# Detecting abnormal gait from an instrumented cane — experimental design

This document describes *what* the experiment does and *why* each choice was made. The measured
outcomes live in [RESULTS.md](RESULTS.md); the code layout and command reference live in
[README.md](README.md).

---

## 1. The question

A walking cane carries an IMU and a downward-facing distance sensor. Given recordings of ordinary
cane-assisted walking, can a model learn what normal gait looks like well enough to flag abnormal
gait it has never seen?

This is framed as **one-class anomaly detection**, not classification. The model is trained only on
normal walking and never shown an anomaly during training. That framing is forced by the data: there
are five kinds of abnormality in the test set and between one and four recordings of each, which is
far too few to learn a decision boundary per class. It is also the more useful framing clinically —
a deployed cane will meet gait deviations nobody recorded in advance.

The mechanism is a **reconstruction autoencoder**. A bottleneck much narrower than the input forces
the network to learn the few degrees of freedom ordinary gait actually has. Reconstruction error is
then the anomaly score: a window the model cannot reproduce is a window unlike anything it was
trained on.

---

## 2. The data

### 2.1 Recording format

A session is one or more 30-second segment files, `rec_XXXXX_segNNN_annY.csv`. `t_ms` restarts in
each segment; `utils/io.py::load_session` stitches a session's segments back into one continuous
recording by offsetting each to start where the previous ended.

| column | meaning |
|---|---|
| `t_ms` | milliseconds since the segment started |
| `ax_mg, ay_mg, az_mg` | acceleration, millig (loaded as g) |
| `gx_mdps, gy_mdps, gz_mdps` | angular rate, millidegrees/s (loaded as deg/s) |
| `dist_mm` | cane base to ground distance, millimetres |
| `event` | `1` on the first sample after a single click during recording |

`dist_mm` and `event` were added by later firmware revisions, so a recording may carry either, both
or neither. Columns are therefore resolved **by name, not position**, which is what lets one loader
read every schema revision. Rows corrupted by serial-logging glitches (wrong field count, unparseable
numbers) are dropped with a warning rather than failing the load.

### 2.2 The distance sensor's three bands

The `dist_mm` reading does not behave like a clean waveform. It quantizes into three bands, whose
thresholds are empirical calibration values for this sensor and mounting (`utils/io.py`):

- **90–110 mm** — resting band. The cane is planted; variation here is sensor jitter.
- **above 118 mm** — the tip has actually lifted away from the ground, i.e. a step.
- **below 90 mm** — a measurement error, not a real reading.

The error floor matters twice over: the step detector clamps to it, and so does the windowing
preprocessor when `dist_mm` is used as a model input.

### 2.3 Split and labels

| directory | contents | role |
|---|---|---|
| `data/train` | 5 sessions, all `ann0` | normal only — everything is fitted here |
| `data/test` | 12 sessions, mixed | labelled, never seen during fitting |

Labels come from the filename annotation, with one case that is not session-level:

| annotation | meaning | window-level treatment |
|---|---|---|
| `ann0` | normal throughout | every window is a negative |
| `ann1` | abnormal throughout | every window is a positive |
| `ann-1` | normal *except* at least one marked moment | only windows overlapping an `event` marker are positives; **all other windows are dropped** |

The `ann-1` rule is the important one. Those windows are dropped rather than counted as normal
because nothing in the recording says where the abnormal stretch ends — calling them normal would
invent a label, and calling them abnormal would invent a different one. They are still *scored*, and
what the detector does on them is reported separately (see §9).

`description.csv` gives the anomaly type per recording. It is used for reporting and grouping only —
nothing in the pipeline branches on it.

---

## 3. Counting steps

The step detector (`exp/step_count.py`) is an experiment in its own right — *how many steps* — and
also a component of the pipeline below, where §4.3 uses detected step times to place windows. Both
roles want the same thing: a detector that finds one event per strike and does not invent events when
the cane is not walking.

### 3.1 The signal

A cane strike is a **broadband mechanical shock**, not a smooth oscillation. The original detector
band-passed the signal around a 0.5–3.5 Hz cadence band, which is the wrong model of the event: a
filter excited by an impulse rings, so one strike emerged as several peaks and the count came out
roughly 3× too high.

What replaces it is the deviation of the acceleration magnitude from gravity, `|acc| - 1g`, across all
three axes at once. Using the magnitude rather than a single axis matters because the cane is not held
at a repeatable angle — a strike is a large shock however it is oriented, but it is only reliably large
in the *combined* magnitude.

### 3.2 The detection rule

A step is a peak in that deviation satisfying three conditions together. Each exists to reject a
different false positive, and none is redundant.

| condition | value | rejects |
|---|---|---|
| **height** | `max(mean + 3.2·std, 0.30 g)` | the walking and handling noise floor |
| **prominence** | ≥ 0.5 × the detection height | a strike's own ringing |
| **refractory** | ≥ 1.15 s since the last accepted step | a second crossing within one strike |

**Height** is set from each recording's own noise floor rather than as a fixed value, because the
noise floor varies with how the cane is handled. The multiplier is the one parameter with no
independent physical anchor, so it was fitted (§3.4). Note the failure mode it has to avoid: a
recording with tremor has a *large* standard deviation, so too high a multiplier puts the threshold
above the very strikes it should be finding — which is what the previous value of 4.3 did, reporting
a single step in recordings like `rec_00018` and `rec_00027`.

The **absolute floor** is what makes the rule safe on a recording that contains no walking at all.
`mean + k·std` is scale-free, so on a motionless recording it happily descends to the sensor's own
noise and reports "steps" in it. Real strikes in `data/` peak at **0.59–1.31 g** of deviation while a
motionless recording peaks around **0.03 g**, so a floor placed between the two rejects the second
without ever binding on the first — it binds on 0 of the 45 scoreable windows.

**Prominence** requires a peak to stand clear of its own surroundings, not merely to clear an absolute
level. This is the condition that distinguishes one shock from the oscillation that follows it: a
rebound can stay above the height threshold while never falling back toward the noise floor, and
height alone cannot tell the two apart.

The **refractory period** caps cadence. Ground-truth step intervals here have a median of **1.97 s**
and a 1st percentile of 0.98 s, so cane cadence tops out near 60 steps/min — nothing like the
150 steps/min the previous 0.4 s gate allowed. That slack was the detector's single largest error
source, and it is a *cadence* assumption, which is the detector's main limitation (§11).

### 3.3 Ground truth from the distance sensor

The `dist_mm` channel gives an independent count that owes nothing to the accelerometer. Because the
reading quantizes into the three bands of §2.2 rather than behaving like a waveform, a peak search
would be fragile against its jitter; instead a step is each **rising crossing** of the 118 mm
threshold, latched with hysteresis. The latch cannot fire again until the reading has come back down
and *stayed* in the resting band for 0.1 s — a bare touch is usually mid-swing jitter rather than the
cane landing.

The two detectors do not mark the same instant. The distance latch fires when the tip **lifts**; the
acceleration peak is the tip **landing**. Measured against detected steps, the offset to the landing
event is a tight −0.12 s (IQR 0.21 s) while the offset to the lift is +0.59 s with an IQR of 1.77 s,
i.e. no relationship at all. They agree on *one event per gait cycle*, which is what a count needs,
so they are compared as counts — but any timing comparison has to be made against the landing.

### 3.4 Choosing the parameters, and scoring them

Accuracy per 30-second window is `1 - |detected - ground_truth| / ground_truth`, clipped at 0.

**Two kinds of window are excluded, and the second one matters.** A window with fewer than
`--min-gt-steps` (default 5) ground-truth steps makes the relative error noisy and is too short a
stretch of gait to say anything. Separately, a window whose ground-truth *cadence* falls below
`--min-gt-cadence` (default 20 steps/min) is excluded because such a count is almost always a
**ground-truth failure rather than slow walking**: the latch needs the tip reading to fall back into
the resting band, so whenever it does not — the sensor lost the ground, the cane was carried, the
baseline drifted — real steps become invisible and the count collapses. These windows are identifiable
without reference to the detector: the tip reads as lifted for 2–32 s at a stretch, against 1.0–1.5 s
on sound windows. Scoring a detector against them measures the sensor, not the detector. 18 of the 63
`ann0` windows are dropped this way; the 45 that remain sit at 22–34 steps/min.

Parameters were chosen by grid search over height multiplier, refractory period and prominence on
those 45 windows. Because 45 windows from 12 sessions is a small set to fit three parameters on, the
reported figure is a **nested leave-one-session-out cross-validation** — the whole parameter tuple is
chosen on 11 sessions and scored on the held-out 12th, so no session contributes to the parameters it
is scored under. The in-sample number is reported alongside it, and the gap between them is the
honest measure of how much the fit is worth (RESULTS §1).

Two independent checks guard against a count that is right for the wrong reason — a detector can miss
one step and invent another and still report the correct total. The first is the **timing-matched**
F1 against ground-truth landings; the second is that the count bias is reported as a signed quantity,
so over- and under-counting cannot cancel in the mean.

---

## 4. Preprocessing

Four stages, all in `utils/windows.py`.

### 4.1 Resample

The firmware's sample interval jitters; the median rate across `data/` is ~67 Hz against a nominal
100 Hz. A window length expressed in seconds would therefore map to a different number of samples in
every recording. Every channel is linearly interpolated onto a uniform grid at `--target-fs`
(default 25 Hz — enough for gait shape and the strike transient, while keeping the input dimension
small, which matters with a few hundred training windows).

Step and event indices are mapped by **nearest sample, not interpolated**: they are impulses, and
interpolating them would smear or erase them.

### 4.2 Window

Windows are `--window-s` seconds long (default 6.0, about three steps — the median interval between
detected steps in `data/` is 1.96 s, cane-assisted gait here being slow and deliberate).

`--anchor` chooses where they are cut:

- **`slide` (default)** — one window every `--hop-s` seconds, regardless of content.
- **`step`** — one window centred on each detected step.

Step anchoring phase-locks windows for free, and it was the original design. It was abandoned because
it makes *how much evidence a recording yields* depend on a step detector — and every step detector
available here degrades on abnormal gait. A shuffled or dragged step produces neither a clean
acceleration shock nor a tip lift past the distance threshold, so the `dist_mm` "ground truth" is no
more reliable on these recordings than the accelerometer is. The consequence is measured in §8.1.

A window is kept only if it fits inside its recording **with `--max-lag-s` of slack on both sides**,
so the alignment stage can shift it by any lag in range and still re-slice real samples. Never zero
padding, and never a window whose length depends on how far it moved.

### 4.3 Align

Windows cut from a continuous recording land at arbitrary points in the gait cycle. If the same tip
strike sits at index 10 in one window and index 40 in another, the model has to learn it separately
at each offset — capacity spent on phase rather than shape. So each window is shifted by a lag before
it is sliced.

`--align-mode` chooses how that lag is found:

| mode | rule | bias |
|---|---|---|
| `none` | lag 0 | none, but phase is unhandled |
| `xcorr` | maximise normalised cross-correlation against a template of normal gait | **searches for the most normal-looking shift** |
| `step` | centre the nearest detected step; unshifted if none is within range | none — placement, not matching |
| `mixed` **(default)** | `step` where a step was detected, `xcorr` only as fallback | fallback only where no step exists |

The bias in `xcorr` is why the other modes exist. The template is built from *normal* gait, so for an
anomalous window "best correlation" means "the pose that looks most normal" — it shrinks precisely
the reconstruction error the score is made of. A step lag cannot do this: the step detector never
sees the template, so it is placement rather than matching.

`mixed` prefers the unbiased branch wherever the evidence for it exists and falls back only where no
step was detected at all. Critically, under sliding anchors this no longer *gates* anything — a
window gets scored either way, so a failed step detection costs a fallback, not a missing window.

Implementation notes:

- The alignment signal is always mean-removed `|acc|`: orientation-independent (the cane is not held
  at a repeatable angle) and where the tip strike is sharpest.
- `xcorr` finds each lag with one `correlate(..., mode="valid")` over an extended slice — a matched
  filter giving every candidate lag at once — then divides by the norm of the sub-window it came
  from, so a lag is not preferred merely because the signal is louder there.
- `--max-lag-s` (default 1.0 s, about half a step interval) is wide enough to fix a mis-anchored
  window and narrow enough that a window cannot lock onto the *neighbouring* step, which would
  collapse the phase structure.
- `--align-iters` (default 3) applies to `xcorr` only. Building its template is chicken-and-egg —
  you need aligned windows to build it and the template to align them — so it starts from the mean of
  unaligned windows, re-estimates every lag, rebuilds, and repeats until the mean lag change drops
  below one sample. `step` and `mixed` never iterate: step lags do not depend on the template, so
  there is no circularity and the template is built in a single pass.

### 4.4 Normalize

Per-channel z-score. Two deliberate choices:

- **Global, not per-window.** Normalizing each window on its own would erase amplitude differences,
  which for anomaly detection are exactly the thing worth noticing.
- **Fitted on training windows only**, then applied frozen.

### 4.5 Channels

`ax, ay, az, gx, gy, gz, |acc|`, plus `dist_mm` by default — eight in total.

The distance channel used to be opt-in, on the grounds that it is the sensor the step detector treats
as ground truth, so feeding it to a model *anchored* on detected steps would be circular. Sliding
anchors largely dissolve that objection: the step detector no longer decides which windows exist, so
the reading is just another sensor. `--no-dist` turns it off.

Readings below the 90 mm error floor are clamped to it, exactly as the step detector clamps them —
otherwise the model would partly be learning to detect sensor dropout. The channel is also disabled
automatically, with a printed note, for any input where some recording predates the sensor.

---

## 5. Model

Two architectures behind `--model`, both taking and returning `(batch, channels, samples)`.

**`conv` (default)** — a 1-D convolutional autoencoder. Strided convolutions downsample to a
`--latent` bottleneck (default 16); the decoder mirrors them, using explicit interpolation back to
each recorded length rather than `ConvTranspose1d`, which avoids off-by-one output sizes for
arbitrary window lengths.

**`mlp`** — a symmetric fully-connected autoencoder over the flattened window, encoder widths from
`--hidden` (default `128,32`).

The conv model is preferred for two reasons:

1. **Parameter economy.** The MLP's first layer alone is `channels × samples × hidden[0]` weights —
   about 279k parameters in total against the conv model's 34k, with only a few hundred training
   windows to fit them.
2. **Shift tolerance.** Convolution is shift-equivariant, so a gait pattern is recognised the same
   way wherever it falls in the window. The MLP must learn each phase separately, which is why it
   leans harder on the alignment stage — and that stage is the one with the bias described in §4.3.

Both use ReLU internally and a **linear output**: the target is z-scored, so squashing it would clip
exactly the large excursions that carry the anomaly signal. Adam, MSE loss, early stopping on
validation loss with the best-validation weights restored.

---

## 6. Training protocol

### 6.1 Train/validation split

The split is on **contiguous time blocks within each training session** — the last `--val-frac` of
each session by time — with a guard band of one full window discarded at the cut.

- Not random at window level: neighbouring windows overlap heavily, so random assignment puts
  near-duplicates on both sides and the validation loss stops measuring generalization.
- Not a whole-session holdout: the training sessions are very uneven (one contributes over half the
  windows, another eleven), so holding one out would either waste most of the data or calibrate on a
  handful of windows.
- The guard band ensures no training window shares a single sample with a validation window.

### 6.2 What is fitted where

Everything is fitted on `data/train` and applied frozen to `data/test`:

| fitted on training data | used frozen on test data |
|---|---|
| alignment template | ✔ |
| per-channel z-score statistics | ✔ |
| network weights | ✔ |
| decision threshold | ✔ |

**No test label enters the pipeline at any point.** A checkpoint stores the weights alongside the
template, the statistics and the threshold, so `--load` scores new recordings through exactly the
pipeline the model was trained on — including its calibrated operating point, rather than deriving a
new one from data that may contain the anomalies.

---

## 7. Scoring and decision

### 7.1 From error to score

`--score-agg` collapses the per-sample squared error of a window into one number:

- **`mean`** — plain mean squared error.
- **`chan_norm` (default)** — each channel's error divided by its median on held-out normal windows
  before averaging, so every channel reports in units of "how unusual is this for this channel"
  rather than letting the channels with the largest natural residual dominate. That scaling is
  fitted on held-out normal training windows and frozen like everything else. It matters far more
  with the distance channel in play than without: `dist_mm` is measured in millimetres and the
  accelerometers in g, so their natural residuals differ by orders of magnitude and a plain mean is
  effectively a weighted vote dominated by whichever channel happens to be noisiest. At seven
  channels the two aggregations tie; at eight, `chan_norm` is decisively better (RESULTS §7).
- **`peak`** — the mean over the worst tenth of time samples, on the theory that a stumble occupies a
  fraction of a window and averaging dilutes it.

### 7.2 The threshold is a declared false-alarm budget

The threshold is the `--threshold-pct` percentile (default 95) of reconstruction error on **held-out
normal training windows**. Stating it this way matters: p95 means "accept a 5% false-alarm rate on
normal gait", which is a design decision made in advance, not a value tuned until the results looked
good. Every report sweeps the budget so the cost of the choice is visible, and every row of that
sweep is reachable without consulting a test label.

### 7.3 Metrics

- **AUPRC** and **ROC-AUC** are threshold-free: they say whether the score *orders* windows
  correctly at all. AUPRC's chance level is the positive rate, not 0.5, so both are reported.
- **F1, accuracy, precision, recall, specificity** at the calibrated threshold: whether the cut lands
  in the right place.
- **Oracle F1** — the best F1 any threshold could achieve. It peeks at test labels and is therefore
  *not* an achievable operating point; it is reported to separate "the score does not separate the
  classes" from "the score separates them but the threshold sits wrong".
- **Session verdict** — a session counts as detected when more than half its scoreable windows are
  flagged. Majority rather than "any window flagged", because on a 20–30 s recording of continuously
  abnormal gait a detector that fires once and goes quiet has not recognised the gait, and "any"
  would call every long recording anomalous on its worst second.

Metrics are implemented over numpy in `utils/metrics.py` rather than pulled from scikit-learn. The
ROC and PR curves step through *groups* of equal scores, so a detector that assigns many windows the
same value is not credited with an ordering it never produced; AUPRC is the step-wise average
precision rather than a trapezoidal integral, which would credit thresholds that do not exist.

---

## 8. Two failure modes this design exists to avoid

### 8.1 Evidence that depends on the thing being detected

Step anchoring yields **196 labelled windows against 290** for sliding anchors, and the shortfall is
not spread evenly: it lands on the abnormal recordings, which yield **4–16 windows each** against
**109** from the normal control — the thinnest being `rec_00019` and `rec_00025` at 4. A recording
tested on four windows has not really been tested — and the reason it yielded four is that the step
detector found less to anchor on there, which is to say it struggled *because* the gait was abnormal.
Any metric computed that way is measured on a test set that under-samples its hardest cases.

The size of this effect depends on the step detector, and it shrank when the detector was retuned
(§3): the same row previously yielded 161 windows with three recordings reduced to *one window each*.
That is worth stating plainly, because it means part of the original case against step anchoring was
really a case against one detector. What does not depend on the detector is the structure of the
failure: under step anchoring, how much evidence a recording yields is a function of how detectable
its gait is, and the recordings that lose the most are the ones the detector is supposed to catch. A
better detector shrinks the effect without removing the dependency.

Sliding anchors give every recording the same coverage per second, whatever its gait looks like.

### 8.2 A preprocessing stage that helps anomalies hide

Cross-correlation alignment optimises each window's similarity to a template of *normal* gait. Applied
to an anomalous window, it is a search for the least anomalous-looking shift, and it shrinks the error
the score is built from. This does not destroy the ranking so much as degrade the operating point —
which is why `xcorr` posts a respectable AUPRC alongside the worst F1 of the four policies
(RESULTS §7.1).

---

## 9. Reporting

`ae-anomaly` writes to `--out-dir`:

| file | contents |
|---|---|
| `metrics.csv` | headline metrics at the calibrated threshold, plus the false-alarm budget swept |
| `sessions.csv` | one row per test session |
| `windows.csv` | one row per scoreable test window |
| `unlabelled_runs.csv` | flagged stretches inside the `ann-1` recording's unlabelled span |
| `overview` | loss curves, score separation, ROC, precision-recall |
| `error_by_type` | error histogram, normal vs each anomaly type |
| `sessions` | per-session flag rate and score spread |
| `traces` | anomaly score against time, per session |
| `examples` | best and worst reconstructions, with per-channel error |

Every plot is written as both `.png` and `.pdf`. **Nothing generated is prose.** The pipeline emits
numbers and figures; the interpretation lives in RESULTS.md, written by hand, so that no claim about
what a result means is produced by the same code that produced the result.

Two diagnostics are worth calling out because they are easy to misread:

- **Alignment sharpness gain** — mean-signal energy after alignment ÷ before. Averaging misaligned
  copies of one pattern cancels it out, so above 1 means the windows now agree on where the pattern
  is. Read it against the anchoring: under `step` anchoring windows start nearly phase-locked so the
  gain is modest, while under `slide` they start at arbitrary phase and the ratio runs into the tens.
  A large number means the baseline was random, not that something is wrong.
- **Alignment branch split** — under `mixed`, how often the correlation fallback was needed. Read it
  as a diagnostic of the *step detector* rather than as a second anomaly score. It moves substantially
  when the detector is retuned: on the current tuning the normal session falls back 7% of the time and
  the abnormal recordings 7–60%, which is informative at the high end (`rec_00025` 60%, `rec_00028`
  56%) but not separating — `rec_00021` falls back as rarely as normal gait. Under the stricter
  earlier tuning the same recordings fell back 43–93%. What it reliably tells you is whether the step
  branch or the fallback is carrying the alignment.

The `ann-1` recording gets its own section, reporting what the detector did on the *unlabelled*
stretches that are excluded from every metric.

---

## 10. Reproduction

```bash
uv sync

# step counting: the detector on one recording, or a directory of sessions
uv run main.py step-count data/rec_00004_seg000_ann0.csv --plot out.png

# step counting: accuracy against the dist_mm ground truth, per window
uv run main.py step-accuracy data --plot-dir out/step_accuracy

# the headline run — bare defaults are the best-measured configuration
uv run main.py ae-anomaly

# the original pipeline, for comparison
uv run main.py ae-anomaly --anchor step --model mlp --align-mode xcorr --latent 8 --no-dist \
  --score-agg mean
```

The CLI runs one configuration per invocation — the shipped defaults, or whatever the flags override.
The multi-variant comparisons in RESULTS.md were produced by a variant runner that has since been
removed, on the principle that the shipped tool should run the configuration that ships and not a
research harness; reproducing those tables means calling `run_ae_anomaly` directly with the overrides
and the seed range each one names.

Seeded throughout (`--seed`, default 0). Runs are deterministic: the same configuration reproduces
bit-identical scores across repeated runs and separate processes.

---

## 11. Known limitations

These bound what any number in RESULTS.md can mean.

1. **One walker.** Five training sessions, one person, one mounting. Nothing here demonstrates
   generalization to a different gait or a differently-mounted cane.
2. **One normal test session.** The false-positive rate rests entirely on `rec_00017`, so specificity
   is measured on a single walk.
3. **Overlapping windows are not independent samples.** The metrics are descriptive of these
   recordings, not estimates with confidence intervals.
4. **Whole-recording labels are coarse.** `ann1` recordings are labelled abnormal end to end,
   including the seconds of ordinary walking that start and finish each one. Some windows counted as
   false negatives are windows of genuinely normal gait inside an abnormal recording — visible as the
   ramp at the start of several traces.
5. **Configuration was partly chosen on the test set.** Latent width, window length, sample rate and
   alignment policy were compared on the same test data the headline numbers come from, so those
   numbers are mildly optimistic. The spread across that grid is the more honest figure.
6. **The distance channel is partly confounded on two sessions.** `rec_00026` and `rec_00028` carry
   ~10% sensor dropout and both are abnormal; the other ten test sessions do not.
7. **`rec_00028` carries one event marker for a recording described as "various falls", plural.**
   Its unlabelled stretches cannot be scored either way.
8. **The step detector assumes this walker's cadence.** Its 1.15 s refractory period caps detection at
   ~52 steps/min. That is comfortable for the 22–34 steps/min of these recordings and merges the
   fastest ~2% of ground-truth steps, but it is a fixed assumption fitted to one person's slow,
   deliberate cane gait and would under-count a faster walker. A cadence-adaptive refractory period is
   the obvious next step, and the one change most likely to matter for a second subject.
9. **The step detector's ground truth is itself a sensor.** 18 of 63 `ann0` windows had to be excluded
   because the distance reading, not the gait, had failed (§3.4). The remaining 45 are the windows
   where both sensors worked, which is a mildly favourable selection — the detector is never scored on
   the recordings where conditions were hardest for the instrument.
