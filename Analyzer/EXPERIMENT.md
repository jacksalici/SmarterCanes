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
what the detector does on them is reported separately (see §8).

`description.csv` gives the anomaly type per recording. It is used for reporting and grouping only —
nothing in the pipeline branches on it.

---

## 3. Preprocessing

Four stages, all in `utils/windows.py`.

### 3.1 Resample

The firmware's sample interval jitters; the median rate across `data/` is ~67 Hz against a nominal
100 Hz. A window length expressed in seconds would therefore map to a different number of samples in
every recording. Every channel is linearly interpolated onto a uniform grid at `--target-fs`
(default 25 Hz — enough for gait shape and the strike transient, while keeping the input dimension
small, which matters with a few hundred training windows).

Step and event indices are mapped by **nearest sample, not interpolated**: they are impulses, and
interpolating them would smear or erase them.

### 3.2 Window

Windows are `--window-s` seconds long (default 6.0, about three steps — the median interval between
detected steps in `data/` is 1.92 s, cane-assisted gait here being slow and deliberate).

`--anchor` chooses where they are cut:

- **`slide` (default)** — one window every `--hop-s` seconds, regardless of content.
- **`step`** — one window centred on each detected step.

Step anchoring phase-locks windows for free, and it was the original design. It was abandoned because
it makes *how much evidence a recording yields* depend on a step detector — and every step detector
available here degrades on abnormal gait. A shuffled or dragged step produces neither a clean
acceleration shock nor a tip lift past the distance threshold, so the `dist_mm` "ground truth" is no
more reliable on these recordings than the accelerometer is. The consequence is measured in §7.1.

A window is kept only if it fits inside its recording **with `--max-lag-s` of slack on both sides**,
so the alignment stage can shift it by any lag in range and still re-slice real samples. Never zero
padding, and never a window whose length depends on how far it moved.

### 3.3 Align

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

### 3.4 Normalize

Per-channel z-score. Two deliberate choices:

- **Global, not per-window.** Normalizing each window on its own would erase amplitude differences,
  which for anomaly detection are exactly the thing worth noticing.
- **Fitted on training windows only**, then applied frozen.

### 3.5 Channels

`ax, ay, az, gx, gy, gz, |acc|`, plus `dist_mm` by default — eight in total.

The distance channel used to be opt-in, on the grounds that it is the sensor the step detector treats
as ground truth, so feeding it to a model *anchored* on detected steps would be circular. Sliding
anchors largely dissolve that objection: the step detector no longer decides which windows exist, so
the reading is just another sensor. `--no-dist` turns it off.

Readings below the 90 mm error floor are clamped to it, exactly as the step detector clamps them —
otherwise the model would partly be learning to detect sensor dropout. The channel is also disabled
automatically, with a printed note, for any input where some recording predates the sensor.

---

## 4. Model

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
   leans harder on the alignment stage — and that stage is the one with the bias described in §3.3.

Both use ReLU internally and a **linear output**: the target is z-scored, so squashing it would clip
exactly the large excursions that carry the anomaly signal. Adam, MSE loss, early stopping on
validation loss with the best-validation weights restored.

---

## 5. Training protocol

### 5.1 Train/validation split

The split is on **contiguous time blocks within each training session** — the last `--val-frac` of
each session by time — with a guard band of one full window discarded at the cut.

- Not random at window level: neighbouring windows overlap heavily, so random assignment puts
  near-duplicates on both sides and the validation loss stops measuring generalization.
- Not a whole-session holdout: the training sessions are very uneven (one contributes over half the
  windows, another eleven), so holding one out would either waste most of the data or calibrate on a
  handful of windows.
- The guard band ensures no training window shares a single sample with a validation window.

### 5.2 What is fitted where

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

## 6. Scoring and decision

### 6.1 From error to score

`--score-agg` collapses the per-sample squared error of a window into one number:

- **`mean`** — plain mean squared error.
- **`chan_norm` (default)** — each channel's error divided by its median on held-out normal windows
  before averaging, so every channel reports in units of "how unusual is this for this channel"
  rather than letting the channels with the largest natural residual dominate. That scaling is
  fitted on held-out normal training windows and frozen like everything else. It matters far more
  with the distance channel in play than without: `dist_mm` is measured in millimetres and the
  accelerometers in g, so their natural residuals differ by orders of magnitude and a plain mean is
  effectively a weighted vote dominated by whichever channel happens to be noisiest. At seven
  channels the two aggregations tie; at eight, `chan_norm` is decisively better (RESULTS §6).
- **`peak`** — the mean over the worst tenth of time samples, on the theory that a stumble occupies a
  fraction of a window and averaging dilutes it.

### 6.2 The threshold is a declared false-alarm budget

The threshold is the `--threshold-pct` percentile (default 95) of reconstruction error on **held-out
normal training windows**. Stating it this way matters: p95 means "accept a 5% false-alarm rate on
normal gait", which is a design decision made in advance, not a value tuned until the results looked
good. Every report sweeps the budget so the cost of the choice is visible, and every row of that
sweep is reachable without consulting a test label.

### 6.3 Metrics

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

## 7. Two failure modes this design exists to avoid

### 7.1 Evidence that depends on the thing being detected

Step anchoring yields **one window each** from three of the abnormal recordings, against 121 from the
normal control. A detector cannot be said to have been tested on a recording it drew one window from
— and the reason it drew one is that the step detector failed there, which is to say it failed
*because* the gait was abnormal. Any metric computed that way is measured on a test set that quietly
excluded its hardest cases.

Sliding anchors give every recording the same coverage per second, whatever its gait looks like.

### 7.2 A preprocessing stage that helps anomalies hide

Cross-correlation alignment optimises each window's similarity to a template of *normal* gait. Applied
to an anomalous window, it is a search for the least anomalous-looking shift, and it shrinks the error
the score is built from. This does not destroy the ranking so much as degrade the operating point —
which is why `xcorr` posts a respectable AUPRC alongside the worst F1 of the four policies (§RESULTS
§4).

---

## 8. Reporting

`ae-anomaly` writes to `--out-dir`:

| file | contents |
|---|---|
| `REPORT.md` | the full generated report |
| `sessions.csv` | one row per test session |
| `windows.csv` | one row per scoreable test window |
| `overview.png` | loss curves, score separation, ROC, precision-recall |
| `error_by_type.png` | error histogram, normal vs each anomaly type |
| `sessions.png` | per-session flag rate and score spread |
| `traces.png` | anomaly score against time, per session |
| `examples.png` | best and worst reconstructions, with per-channel error |

Two diagnostics are worth calling out because they are easy to misread:

- **Alignment sharpness gain** — mean-signal energy after alignment ÷ before. Averaging misaligned
  copies of one pattern cancels it out, so above 1 means the windows now agree on where the pattern
  is. Read it against the anchoring: under `step` anchoring windows start nearly phase-locked so the
  gain is modest, while under `slide` they start at arbitrary phase and the ratio runs into the tens.
  A large number means the baseline was random, not that something is wrong.
- **Alignment branch split** — under `mixed`, how often the correlation fallback was needed. This
  turned out to be a diagnostic in its own right, and one that needs no model at all: how often a
  recording needs the fallback is a readout of how much its gait still looks like stepping.

The `ann-1` recording gets its own section, reporting what the detector did on the *unlabelled*
stretches that are excluded from every metric.

---

## 9. Reproduction

```bash
uv sync

# the headline run — bare defaults are the best-measured configuration
uv run main.py ae-anomaly

# with the pipeline comparison, ranked against seed noise
uv run main.py ae-anomaly --ablation --ablation-seeds 3

# the original pipeline, for comparison
uv run main.py ae-anomaly --anchor step --model mlp --align-mode xcorr --latent 8 --no-dist \
  --score-agg mean
```

The ablation walks a **cumulative path** from that original configuration to the current default,
adding one change per row, and its last row is asserted to equal the shipped defaults.

Seeded throughout (`--seed`, default 0). Ablation variants pin every field they vary, so the
comparison cannot silently inherit the caller's configuration.

---

## 10. Known limitations

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
