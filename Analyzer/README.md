# SmartCane Analyzer

Analyzes IMU logs recorded from an instrumented walking cane.

## Architecture

- `main.py` — CLI entry point, dispatches to experiments; owns all printing and plotting
- `utils/` — shared code
  - `io.py` — robust CSV loader → `ImuRecording` (name-based columns, tolerates serial-glitch rows)
  - `windows.py` — windowing preprocessing: resampling, step-anchored *or* sliding windowing, four alignment policies, normalization
  - `metrics.py` — ROC, precision-recall, AUPRC and threshold metrics over numpy
- `exp/` — one module per experiment (`ae_report.py` is the reporting half of `ae_anomaly.py`)
- `out/` — generated plots and model checkpoints (git-ignored)

The recordings themselves live outside the Analyzer, in `../Dataset/`:

- `Dataset/data/` — every `rec_*.csv` recording, flat, no subfolders
- `Dataset/split.csv` — `filename,split` table labelling each recording `train`, `test` or `walk`.
  `train` (normal gait only) and `test` (labelled, normal and anomalous alike) are the pair
  `ae-anomaly` fits and evaluates on; `walk` is the rest of the pool, with no role in that protocol.

  `step-count`, `step-accuracy` and `step-ae` take `--split` to choose which of `Dataset/data` they
  run over: the default, `normal`, is every `ann0`/legacy recording regardless of its `split.csv`
  label — `train`, `walk`, and the handful of `test` recordings (such as the held-out normal control
  session) that are `ann0` too — since ordinary gait analysis has no use for the labelled anomalies
  `ae-anomaly` exists to score. `all` is the whole pool, anomalies included; a comma-separated subset
  of `train`, `test`, `walk` selects by `split.csv` label directly, anomalies and all.

## Experiments

- **step-count** (`exp/step_count.py`) — counts steps as sharp isolated excursions of the acceleration
  magnitude away from 1 g, with the height threshold set from each recording's own noise floor
  (mean + k·std of `|acc|-1g`). A cane strike is a broadband mechanical shock rather than a smooth
  oscillation, so the earlier 0.5–3.5 Hz cadence band-pass smeared one impulse into a ringing filter
  response and over-counted by roughly 3×. A strike must also clear a 1.15 s refractory period, stand
  out from its surroundings by half the detection height, and exceed an absolute 0.30 g floor — the
  first two suppress a strike's own ringing, the third keeps a motionless recording from reporting
  steps at all. Cross-checked against a ground-truth count derived from the distance sensor with a
  three-band hysteresis latch (`dist_mm`: 90–110 mm resting, >118 mm tip lifted, <90 mm sensor error).

- **step-accuracy** (`exp/step_accuracy.py`) — scores the detected count against that ground truth,
  per 30-second window. Windows are dropped when the ground truth cannot support a comparison: fewer
  than `--min-gt-steps` steps, or a ground-truth cadence below `--min-gt-cadence`, which is how a
  distance-sensor failure shows up (if the tip reading never falls back into the resting band the latch
  never re-arms, so real steps go uncounted). Scoring against those windows measures the sensor rather
  than the detector. Mean accuracy is **0.952** over the 45 scoreable windows, 0.934 under nested
  leave-one-session-out cross-validation — see [RESULTS.md](RESULTS.md) §1.

- **step-ae** (`exp/step_ae.py`) — an autoencoder over aligned multi-step windows, trained on
  normal data only, using reconstruction error as an anomaly score. Where step-count answers *how many*
  steps, this asks what a step *looks like*. See below.

- **ae-anomaly** (`exp/ae_anomaly.py`, reporting in `exp/ae_report.py`) — the same autoencoder put
  through the protocol a detector actually has to survive: everything is fitted on the `train` split
  (normal gait only) and applied frozen to the `test` split, which carries labels the model never sees.
  Reports AUPRC, ROC-AUC, F1 and accuracy per window plus a per-session verdict, and writes a
  standalone report with plots. See below.

## step-ae pipeline

Four preprocessing stages (`utils/windows.py`), then the model (`exp/step_ae.py`).

1. **Resample.** The firmware's sample interval jitters — the median rate across the dataset is ~67 Hz
   against a nominal 100 Hz — so a window length in seconds would otherwise map to a different number
   of samples in every recording. Every channel is linearly interpolated onto a uniform grid at
   `--target-fs` (default 25 Hz, enough for gait shape and the strike transient while keeping the input
   dimension small). Step and event indices are mapped by nearest sample; event markers are impulses,
   so interpolating them would smear or erase them.

2. **Window.** Windows of `--window-s` seconds (default 6.0 ≈ three steps: the median interval between
   detected steps in the dataset is 1.92 s, cane-assisted gait here being slow and deliberate — a plain
   walking cadence would put three steps nearer 3 s), cut every `--hop-s` seconds under the default
   `--anchor slide`. `--anchor step` instead centres one window on each detected step (with
   `--step-hop` to take every Nth); it phase-locks windows for free but ties how much evidence a
   recording yields to a step detector that fails on abnormal gait, which is why it is no longer the
   default — see **ae-anomaly** below. A window is kept only if it fits inside its recording with
   `--max-lag-s` of slack on both sides, so the alignment stage can shift it by any lag in range and
   still re-slice real samples — never zero padding, and never a window whose length depends on how far
   it moved.

3. **Align.** Each window is placed into a common phase by one of four policies (`--align-mode`, see
   the table under **ae-anomaly**); the default `mixed` centres the nearest detected step and falls
   back to cross-correlation only where no step was found. The point is that the same part of the gait
   cycle lands at the same index in every window, so the model doesn't spend capacity modelling phase
   as if it were signal. The alignment signal is mean-removed `|acc|` — orientation-independent, since
   the cane is not held at a repeatable angle, and where the tip strike is sharpest. For `xcorr` the
   template is refined iteratively
   (`--align-iters`, default 3): start from the mean alignment signal at the raw anchors, re-estimate
   every lag against it, rebuild the template from the shifted windows, repeat until the mean lag change
   is under one sample. That iteration is what breaks the chicken-and-egg between "where is the pattern"
   and "where is each window relative to it"; it converges in a couple of passes because anchoring
   already puts windows within a fraction of a cycle of each other.

   Each lag is found with one `correlate(..., mode="valid")` over an extended slice — sliding the
   template across the signal as a matched filter, giving every candidate lag at once — then divided by
   the norm of the sub-window it came from, so a lag isn't preferred merely because the signal is louder
   there. The search is capped at `±--max-lag-s` (default 1.0 s, about half a step interval): wide enough
   to fix a mis-anchored window, narrow enough that a window can't align onto the *neighbouring* step and
   collapse the phase structure. The printed sharpness gain is the ratio of mean-signal energy after
   alignment to before — averaging misaligned copies of one pattern cancels it out, so above 1 means
   the windows now agree on where the pattern is. Read it against the anchoring: under `--anchor step`
   the windows start nearly phase-locked, so the gain is modest (1.45× on the `train` split), while under
   `--anchor slide` they start at arbitrary phase and the "before" mean almost cancels, so the ratio
   runs into the tens. A large number there means the baseline was random, not that something is wrong.

4. **Normalize.** Per-channel z-score, with statistics computed over the *training* windows only, and
   global rather than per-window: normalizing each window on its own would erase amplitude differences,
   which for anomaly detection are exactly the thing worth noticing.

**Channels.** `ax, ay, az, gx, gy, gz, |acc|`, plus `dist_mm` — the cane tip's height above the
ground — as an eighth channel by default. It measurably helps (see **ae-anomaly**), and the old
objection to it (that it is the step detector's ground truth) mostly dissolved once windows stopped
being anchored on detected steps. `--no-dist` scores from the IMU alone. Because `dist_mm` was added by
a later firmware revision, the channel is switched off automatically — with a printed note — for any
input where some recording lacks the column, so older data still loads.

**Model.** Either of two autoencoders, chosen with `--model`. The default `mlp` is a symmetric MLP
autoencoder over the flattened `(channels × samples)` window — encoder
widths from `--hidden` (default `128,32`) down to a `--latent` bottleneck (default 16), mirrored back,
ReLU between and a linear output (the target is z-scored, so squashing it would clip exactly the large
excursions that carry the anomaly signal). Adam, MSE, early stopping on validation loss.

`conv` (the default) is a 1-D convolutional autoencoder instead: strided convolutions down to the same
bottleneck, mirrored by interpolation back up. It is far smaller for the same window — about 34k
parameters against 279k — and shift-tolerant, which is what lets it cope with whatever residual phase
the alignment policy leaves behind.

Trained on normal windows only, so a bottleneck far narrower than the input forces it to learn the few
degrees of freedom ordinary gait actually has. Reconstruction error is then the anomaly score, and the
decision threshold is a high percentile (`--threshold-pct`, default 95) of the error on *held-out
normal* windows — calibrated on what normal looks like rather than on any assumption about anomalies.
A model loaded with `--load` keeps the threshold stored in its checkpoint rather than re-deriving one
from the recordings being scored, which would calibrate the operating point on data that may contain
the very anomalies it is meant to catch.

The train/validation split is **by session**, not by window: consecutive windows overlap heavily, so a
window-level split would put near-duplicates on both sides and make the validation loss meaningless.
The template and the normalization statistics are likewise fitted on normal training windows and then
applied frozen, so nothing about the held-out windows leaks into the preprocessing. A checkpoint stores
the weights alongside the template and statistics, so `--load` scores new recordings through exactly the
pipeline the model was trained on.

**Labels.** Two label sources are wired up behind `--normal-by`:

- `event` — a window containing an `event == 1` sample (a single click during recording, a
  user-flagged moment of interest) is held out as a candidate anomaly. Recordings with no `event`
  column are all normal, so older data stays usable.
- `ann` — the filename annotation, following the convention the recordings actually use rather than a
  flat allow-list:

  | annotation | meaning | treatment |
  |---|---|---|
  | `ann0` | normal throughout | every window is training data |
  | `ann1` | abnormal throughout | every window is a candidate |
  | `ann-1` | normal *except* at least one marked moment | normal per window, unless it contains an `event` marker |
  | none | legacy file | normal, so old recordings stay usable |

  `--normal-ann` (default `0`) lists the annotations that are normal *end to end*. The `ann-1` rule is
  fixed by what that annotation means, not by the flag, so it applies whether or not `-1` is in the
  list — an `ann-1` recording is mostly ordinary walking, and discarding it wholesale would throw away
  usable training data. An `ann-1` recording carrying no `event` column has no way to say where the
  abnormal stretch is, so it becomes a candidate rather than being silently trusted.

  On the `test` split this yields 40 candidates from 187 step-anchored windows: the 37 from the `ann1`
  sessions, plus the 3 `ann-1` windows holding rec_00028's marker, with that recording's other 26
  windows kept as normal.

**Caveat on scale.** The dataset currently yields ~760 step-anchored windows, still fewer than the input
dimension (7 × 150 = 1050).
The bottleneck, `--weight-decay` and the by-session validation split are all there to keep that honest;
the train/validation gap in the loss curves is the thing to check before trusting a score.

## Commands

All four commands default to `../Dataset/data`, so the directory argument below can be omitted; it is
shown for clarity. `--split` (on `step-count`, `step-accuracy`, `step-ae`) narrows that directory:
`normal` (default) is every `ann0`/legacy recording; `all` is the whole pool, anomalies included; a
comma-separated subset of `train`, `test`, `walk` selects by `Dataset/split.csv` label.

```bash
# step counting: one session
uv run main.py step-count ../Dataset/data/rec_00004_seg000_ann1.csv --plot out.png

# step counting: every recording in a directory, grouped into sessions
uv run main.py step-count ../Dataset/data --plot-dir out/

# step counting over the whole dataset, labelled anomalies included
uv run main.py step-count --split all --plot-dir out/

# step-count accuracy against the dist_mm ground truth, per window
uv run main.py step-accuracy ../Dataset/data --plot-dir out/step_accuracy

# autoencoder: train on all recordings and score them
uv run main.py step-ae ../Dataset/data --plot-dir out/step_ae

# a different window length, and windows anchored on every other detected step
uv run main.py step-ae ../Dataset/data --window-s 3.5 --anchor step --step-hop 2 --plot-dir out/step_ae

# use the filename annotation as the label source instead of the event marker
uv run main.py step-ae ../Dataset/data --normal-by ann

# score with a saved model instead of training
# (reuses its template, statistics and calibrated threshold)
uv run main.py step-ae ../Dataset/data --load out/step_ae/model.pt --plot-dir out/step_ae
```

The `step-ae` plot has five panels: the alignment signal's mean ± std before and after alignment (does
alignment sharpen the common pattern?), the lag histogram (lags piling up at the bounds means
`--max-lag-s` is too tight), the loss curves, the reconstruction-error distribution against the
threshold, and the worst-reconstructed window against its reconstruction.

## ae-anomaly: evaluating the detector

`step-ae` trains and scores one pool of recordings, which cannot answer whether the detector works —
for that the fit and the measurement have to be separated. `ae-anomaly` fits the alignment template,
the normalization statistics, the weights and the decision threshold on the `train` split and applies all
four frozen to the `test` split. No test label enters the pipeline at any point.

**Labels** come from the filename annotation. `ann0` is normal and every window is a negative; `ann1`
is abnormal gait throughout and every window is a positive; `ann-1` is a normal recording containing
at least one abnormal moment located only by the in-recording `event` click, so only windows
overlapping a marker are positives and the rest are *dropped* rather than called normal — nothing
records where the abnormal stretch ends. `description.csv` supplies the anomaly type per recording,
for the report only.

**Metrics** are in `utils/metrics.py` (numpy, no scikit-learn): AUPRC and ROC-AUC are threshold-free
and say whether the score orders windows correctly at all; F1 and accuracy at the calibrated threshold
say whether the cut lands in the right place. AUPRC's chance level is the positive rate, not 0.5, so
the report prints both. The oracle F1 — the best any threshold could do — is reported alongside, to
separate a ranking failure from an operating-point failure.

### Two fixes this experiment forced

**Step anchoring had to go.** Cutting one window per detected step makes the amount of evidence
gathered from a recording depend on a step detector, and every step detector here degrades on abnormal
gait: a shuffled or dragged step produces neither a clean acceleration shock nor a tip lift past the
`dist_mm` threshold, so the `dist_mm` "ground truth" is no more trustworthy on these recordings than
the accelerometer is. How badly that bites depends on how the detector is tuned, which is the deeper
objection: the size of the test set becomes a function of a threshold nobody set with evaluation in
mind. On the current tuning, step anchoring yields 4–16 windows from each abnormal recording against
109 from the normal rec_00017, and 196 labelled windows against 290 for sliding. `--anchor
slide` (the default here) cuts a window every `--hop-s` seconds regardless of content, so coverage is
uniform in time and identical across classes.

**Alignment had to change policy.** Cross-correlating each window against the *normal* template and
keeping the best-matching shift lets an abnormal window search for the pose that looks most normal,
shrinking the very error the score is built from. `--align-mode` offers four policies:

| mode | how a window is placed | bias |
|---|---|---|
| `none` | left at its anchor | none, but phase is unhandled |
| `xcorr` | best correlation against the normal template | searches for the most normal-looking shift |
| `step` | nearest detected step centred; unshifted if there is none | none — placement, not matching |
| `mixed` | `step` where a step was detected, `xcorr` only as fallback | fallback only where no step exists |

`mixed` (the default) is the compromise: step placement is preferred wherever the evidence for it
exists, because it has no bias — the step detector never sees the template — and correlation covers
only the windows where no step was detected at all. Crucially this no longer *gates* anything: with
sliding anchors a window is scored either way, so a failed step detection costs a fallback rather than
a missing window.

The branch split is worth watching, though how much it says depends on the step detector's tuning. On
the current tuning 94% of training windows take the step branch and 93% of the normal test session's
do, while the abnormal recordings fall back between 7% and 60% — informative at the high end
(rec_00025 60%, rec_00028 56%) but not a clean readout: rec_00021 falls back only 7%, the same as
normal gait. Under an earlier, stricter detector tuning the same recordings fell back 43–93% of the
time, so treat the split as a diagnostic of the *detector*, not a second anomaly score.

The conv model is also the smaller one — about 34k parameters against 279k for the MLP at the default
geometry, which matters when there are a few hundred training windows — and being shift-tolerant it
copes with whatever residual phase each policy leaves behind.

**On reading the pipeline comparisons in [RESULTS.md](RESULTS.md).** The better variants differ by a
few thousandths of AUPRC, which is the same order as the spread across random seeds (±0.005 over five
seeds), so no single run ranks them. Those comparisons were measured with a variant runner that has
since been removed — the CLI now runs only the shipped configuration — so reproducing them means
driving `run_ae_anomaly` directly with the overrides and seeds each table names.

### The distance channel

`dist_mm` — the cane tip's height above the ground — is the eighth input channel, **on by default**.
It used to be opt-in because it is the sensor the step detector treats as ground truth, so feeding it
to a model *anchored* on detected steps would be circular. Under sliding anchors that objection is much
weaker: the step detector no longer decides which windows exist, so the reading is just another sensor.
`--no-dist` turns it off, and it is disabled automatically for recordings that predate the sensor.

It helps, consistently and by more than seed noise, under every alignment policy. Two caveats travel
with it. Readings below the sensor error floor (`DIST_ERROR_FLOOR_MM`, 90 mm) are measurement failures
rather than a tip on the ground, and are clamped exactly as the step detector clamps them. And that
dropout is not evenly spread — it is ~11% of rec_00026 and ~10% of rec_00028 and essentially absent
everywhere else, both abnormal recordings, so for those two sessions part of the gain could be the
model spotting a failing sensor rather than a failing gait. The other ten test sessions are clean, so
the improvement there is genuine tip-height information.

### Choosing the threshold

The threshold is a percentile of the reconstruction error on held-out normal *training* windows, which
makes it a declared false-alarm budget rather than something tuned on results: `--threshold-pct 95`
means "accept a 5% false-alarm rate on normal gait". The report sweeps the budget so the cost of the
choice is visible; every row of that table is reachable without consulting a test label.

The train/validation split is on **contiguous time blocks within each training session**, with a
one-window guard band discarded at the cut. Not random: neighbouring windows overlap heavily, so
random assignment puts near-duplicates on both sides. Not whole-session either: the training sessions
are uneven (rec_00009 alone is over half the windows, rec_00012 contributes eleven), so holding one
out would either waste most of the data or calibrate the threshold on a handful of windows. Splitting
inside every session keeps the validation set representative of all of them.

### Output

`--out-dir` (default `out/ae_anomaly/`) receives CSVs and plots, and nothing else — there is no
generated prose, so [RESULTS.md](RESULTS.md) is the only place a claim is made about what any of it
means.

| output | contents |
|---|---|
| `metrics.csv` | headline metrics at the calibrated threshold, plus the false-alarm budget swept |
| `sessions.csv` | one row per test session |
| `windows.csv` | one row per scoreable test window |
| `unlabelled_runs.csv` | flagged stretches inside the `ann-1` recording's unlabelled span |
| `overview` | loss curves, score separation, ROC, precision-recall |
| `error_by_type` | reconstruction error, normal against each anomaly type |
| `sessions` | per-session flag rate and score spread |
| `traces` | anomaly score against time for every test session |
| `examples` | best and worst reconstructions, with per-channel error |

Every plot is written twice, as `.png` and as `.pdf` — raster to look at, vector to print.

```bash
# the headline run - bare defaults are the best-measured configuration
uv run main.py ae-anomaly

# the original pipeline, for comparison
uv run main.py ae-anomaly --anchor step --model mlp --align-mode xcorr --latent 8 --no-dist --score-agg mean

# score from the IMU alone, without the cane-tip distance sensor
uv run main.py ae-anomaly --no-dist

# a stricter false-alarm budget
uv run main.py ae-anomaly --threshold-pct 99
```

## CSV Format

A session is stored as one or more 30-second segment files, `rec_XXXXX_segNNN_annY.csv`, where `XXXXX`
is the session index, `NNN` the zero-based segment number, and `Y ∈ {-1, 0, 1}` the stop annotation
(shared by every segment of the session). `t_ms` restarts from 0 in each segment; consecutive segments
of one session are read back-to-back as a single continuous recording by `load_session`.

```
t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps,dist_mm,event
0,100,50,980,10,20,5,-1,0
10,102,48,982,12,18,6,842.5,0
20,101,49,981,11,19,5,843.1,1
```

- `t_ms` — milliseconds since the segment started
- `ax/ay/az` — acceleration in millig (loaded as g)
- `gx/gy/gz` — rotation in millidegrees/sec (loaded as deg/s)
- `dist_mm` — distance in millimetres from the Modulino Distance sensor; `-1` if no reading is available yet
- `event` — `1` on the first row sampled after a single click during recording, `0` otherwise

`dist_mm` and `event` were added by later firmware revisions, so a recording may carry either, both or
neither; `utils/io.py` resolves columns by name and exposes `has_dist` / `has_event`.

Setup: `uv sync`, then run commands with `uv run`.
