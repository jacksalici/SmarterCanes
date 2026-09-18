# Detecting abnormal gait from an instrumented cane — experimental design

This document describes *what* the experiment does and *why* each choice was made. The code layout and
command reference live in [README.md](README.md).

---

## 1. The question

A walking cane carries an IMU and a downward-facing distance sensor. Given recordings of ordinary
cane-assisted walking, can a model learn what normal gait looks like well enough to flag abnormal gait it
has never seen?

This is framed as **one-class anomaly detection**, not classification. The model is trained only on
normal walking and never shown an anomaly during training. That framing is forced by the data: there are
five kinds of abnormality in the test set and between one and four recordings of each, far too few to
learn a decision boundary per class. It is also the more useful framing clinically — a deployed cane will
meet gait deviations nobody recorded in advance.

The mechanism is a **reconstruction autoencoder**. A bottleneck much narrower than the input forces the
network to learn the few degrees of freedom ordinary gait actually has. Reconstruction error is then the
anomaly score: a window the model cannot reproduce is a window unlike anything it was trained on.

---

## 2. The data

### 2.1 Recording format

A session is one or more 30-second segment files, `rec_XXXXX_segNNN_annY.csv`. `t_ms` restarts in each
segment; `utils/io.py::load_session` stitches a session's segments back into one continuous recording.

| column | meaning |
|---|---|
| `t_ms` | milliseconds since the segment started |
| `ax_mg, ay_mg, az_mg` | acceleration, millig (loaded as g) |
| `gx_mdps, gy_mdps, gz_mdps` | angular rate, millidegrees/s (loaded as deg/s) |
| `dist_mm` | cane base to ground distance, millimetres |
| `event` | `1` on the first sample after a single click during recording |

`dist_mm` and `event` were added by later firmware revisions, so a recording may carry either, both or
neither. Columns are resolved **by name, not position**, which is what lets one loader read every schema
revision. Rows corrupted by serial-logging glitches are dropped with a warning rather than failing the
load.

### 2.2 The distance sensor's three bands

The `dist_mm` reading quantizes into three bands, whose thresholds are empirical calibration values for
this sensor and mounting (`utils/io.py`):

- **90–110 mm** — resting band. The cane is planted; variation here is sensor jitter.
- **above 118 mm** — the tip has lifted away from the ground, i.e. a step.
- **below 90 mm** — a measurement error, not a real reading.

The error floor matters twice: the step detector clamps to it, and so does the windowing preprocessor
when `dist_mm` is used as a model input.

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
| `ann-1` | normal *except* at least one marked moment | only windows overlapping an `event` marker are positives; all other windows are dropped |

The `ann-1` rule drops rather than labels the unmarked windows because nothing in the recording says
where the abnormal stretch ends — calling them normal or abnormal would both invent a label. They are
still scored, and reported separately (§9).

`description.csv` gives the anomaly type per recording, used for reporting and grouping only.

---

## 3. Counting steps

The step detector (`exp/step_count.py`) is both an experiment in its own right and a component of the
pipeline below, where §4.2 uses detected step times to place windows.

### 3.1 The signal

A cane strike is a broadband mechanical shock, not a smooth oscillation, so the detector works on the
deviation of the acceleration magnitude from gravity, `|acc| - 1g`, across all three axes at once. Using
the magnitude rather than a single axis matters because the cane is not held at a repeatable angle.

### 3.2 The detection rule

A step is a peak in that deviation satisfying three conditions together, each rejecting a different
false positive:

| condition | value | rejects |
|---|---|---|
| **height** | `max(mean + 3.2·std, 0.30 g)` | the walking and handling noise floor |
| **prominence** | ≥ 0.5 × the detection height | a strike's own ringing |
| **refractory** | ≥ 1.15 s since the last accepted step | a second crossing within one strike |

**Height** is set from each recording's own noise floor rather than a fixed value, since the floor varies
with how the cane is handled. The **absolute floor** keeps the rule safe on a recording with no walking
at all — `mean + k·std` is scale-free and would otherwise descend to the sensor's own noise on a
motionless recording. Real strikes in `data/` peak at 0.59–1.31 g of deviation, a motionless recording
around 0.03 g. **Prominence** distinguishes one shock from the oscillation that follows it, since a
rebound can stay above the height threshold without falling back toward the noise floor. The
**refractory period** caps cadence at what ground-truth step intervals here support (median 1.97 s, 1st
percentile 0.98 s).

### 3.3 Ground truth from the distance sensor

The `dist_mm` channel gives an independent count. Because the reading quantizes into the three bands of
§2.2, a step is each rising crossing of the 118 mm threshold, latched with hysteresis — the latch cannot
fire again until the reading has returned to and stayed in the resting band for 0.1 s.

The two detectors do not mark the same instant: the distance latch fires when the tip lifts, the
acceleration peak when it lands. They agree on one event per gait cycle, which is what a count needs, so
they are compared as counts; any timing comparison has to be made against the landing.

### 3.4 Choosing the parameters, and scoring them

Accuracy per 30-second window is `1 - |detected - ground_truth| / ground_truth`, clipped at 0.

Two kinds of window are excluded. A window with fewer than `--min-gt-steps` (default 5) ground-truth
steps is too short a stretch of gait to score. A window whose ground-truth cadence falls below
`--min-gt-cadence` (default 20 steps/min) is excluded because such a count is almost always a
ground-truth failure rather than slow walking — the latch needs the tip reading to fall back into the
resting band, so if it doesn't (sensor lost the ground, cane carried, baseline drift), real steps become
invisible and the count collapses. 18 of the 63 `ann0` windows are dropped this way; the remaining 45 sit
at 22–34 steps/min.

Parameters were chosen by grid search over height multiplier, refractory period and prominence on those
45 windows, reported as a nested leave-one-session-out cross-validation — the parameter tuple is chosen
on 11 sessions and scored on the held-out 12th. The in-sample number is reported alongside it, and the
gap between them is the honest measure of how much the fit is worth.

Two independent checks guard against a count that is right for the wrong reason: the timing-matched F1
against ground-truth landings, and reporting the count bias as a signed quantity so over- and
under-counting cannot cancel in the mean.

---

## 4. Preprocessing

Four stages, all in `utils/windows.py`.

### 4.1 Resample

The firmware's sample interval jitters — median rate across `data/` ~67 Hz against a nominal 100 Hz —
so a window length in seconds would map to a different sample count in every recording. Every channel is
linearly interpolated onto a uniform grid at `--target-fs` (default 25 Hz). Step and event indices are
mapped by nearest sample, not interpolated, since they are impulses.

### 4.2 Window

Windows are `--window-s` seconds long (default 6.0, about three steps at this dataset's cadence).
`--anchor` chooses where they are cut:

- **`slide` (default)** — one window every `--hop-s` seconds, regardless of content.
- **`step`** — one window centred on each detected step.

Step anchoring phase-locks windows for free but makes how much evidence a recording yields depend on a
step detector that degrades on abnormal gait (§8.1). A window is kept only if it fits inside its
recording with `--max-lag-s` of slack on both sides, so the alignment stage can shift it without zero
padding.

### 4.3 Align

Windows cut from a continuous recording land at arbitrary points in the gait cycle; each is shifted by a
lag before slicing so the model isn't spending capacity on phase rather than shape. `--align-mode`
chooses how the lag is found:

| mode | rule | bias |
|---|---|---|
| `none` | lag 0 | none, but phase is unhandled |
| `xcorr` | maximise normalised cross-correlation against a template of normal gait | searches for the most normal-looking shift |
| `step` | centre the nearest detected step; unshifted if none is within range | none — placement, not matching |
| `mixed` (default) | `step` where a step was detected, `xcorr` only as fallback | fallback only where no step exists |

`xcorr`'s bias is why the other modes exist: its template is built from normal gait, so for an anomalous
window "best correlation" means "the pose that looks most normal," shrinking the reconstruction error the
score is built from. A step lag can't do this, since the step detector never sees the template. `mixed`
prefers the unbiased branch wherever the evidence exists and falls back only where no step was detected;
under sliding anchors this never gates a window out — a failed step detection costs a fallback, not a
missing window.

Implementation notes:

- The alignment signal is always mean-removed `|acc|`: orientation-independent and where the tip strike
  is sharpest.
- `xcorr` finds each lag with one `correlate(..., mode="valid")` over an extended slice, divided by the
  norm of the sub-window it came from, so a lag isn't preferred merely because the signal is louder there.
- `--max-lag-s` (default 1.0 s, about half a step interval) is wide enough to fix a mis-anchored window
  and narrow enough that a window can't lock onto the neighbouring step.
- `--align-iters` (default 3, `xcorr` only) starts from the mean of unaligned windows, re-estimates every
  lag, rebuilds the template, and repeats until the mean lag change drops below one sample.

### 4.4 Normalize

Per-channel z-score, global rather than per-window (so amplitude differences survive, which anomaly
detection depends on), fitted on training windows only and applied frozen.

### 4.5 Channels

`ax, ay, az, gx, gy, gz, |acc|`, plus `dist_mm` by default — eight in total. Readings below the 90 mm
error floor are clamped to it, matching the step detector; the channel disables itself automatically for
any input where some recording predates the sensor. `--no-dist` turns it off.

---

## 5. Model

Two architectures behind `--model`, both taking and returning `(batch, channels, samples)`.

**`conv` (default)** — a 1-D convolutional autoencoder. Strided convolutions downsample to a `--latent`
bottleneck (default 16); the decoder mirrors them with explicit interpolation back to each recorded
length rather than `ConvTranspose1d`, avoiding off-by-one output sizes for arbitrary window lengths.

**`mlp`** — a symmetric fully-connected autoencoder over the flattened window, encoder widths from
`--hidden` (default `128,32`).

`conv` is preferred: it has far fewer parameters for the same window (~34k against ~279k for `mlp`,
against only a few hundred training windows), and convolution is shift-equivariant, so it recognises a
gait pattern the same way wherever it falls in the window — `mlp` must learn each phase separately and so
leans harder on the alignment stage.

Both use ReLU internally and a linear output (the target is z-scored, so squashing would clip the large
excursions that carry the anomaly signal). Adam, MSE loss, early stopping on validation loss with the
best-validation weights restored.

---

## 6. Training protocol

### 6.1 Train/validation split

The split is on contiguous time blocks within each training session — the last `--val-frac` of each
session by time — with a guard band of one full window discarded at the cut. Not random at window level,
since neighbouring windows overlap heavily and random assignment would put near-duplicates on both sides.
Not a whole-session holdout either, since the training sessions are uneven (one contributes over half the
windows, another eleven) and holding one out would waste most of the data or calibrate on a handful of
windows.

### 6.2 What is fitted where

Everything is fitted on `data/train` and applied frozen to `data/test`: the alignment template,
per-channel z-score statistics, network weights, and decision threshold. No test label enters the
pipeline at any point. A checkpoint stores the weights alongside the template, the statistics and the
threshold, so `--load` scores new recordings through exactly the pipeline the model was trained on.

---

## 7. Scoring and decision

### 7.1 From error to score

`--score-agg` collapses the per-sample squared error of a window into one number:

- **`mean`** — plain mean squared error.
- **`chan_norm` (default)** — each channel's error divided by its median on held-out normal windows
  before averaging, so no channel dominates purely because its natural residual is larger (`dist_mm` is
  in millimetres, the accelerometers in g). Fitted on held-out normal training windows and frozen. At
  seven channels the two aggregations tie; at eight (with `dist_mm`), `chan_norm` is decisively better.
- **`peak`** — the mean over the worst tenth of time samples, on the theory that a stumble occupies a
  fraction of a window and averaging dilutes it.

### 7.2 The threshold is a declared false-alarm budget

The threshold is the `--threshold-pct` percentile (default 95) of reconstruction error on held-out normal
training windows: p95 means "accept a 5% false-alarm rate on normal gait," a design decision made in
advance rather than a value tuned until results looked good.

### 7.3 Metrics

- **AUPRC** and **ROC-AUC** are threshold-free: they say whether the score orders windows correctly at
  all. AUPRC's chance level is the positive rate, not 0.5.
- **F1, accuracy, precision, recall, specificity** at the calibrated threshold say whether the cut lands
  in the right place.
- **Oracle F1** — the best F1 any threshold could achieve. It peeks at test labels and is not an
  achievable operating point; it separates "the score does not separate the classes" from "the score
  separates them but the threshold sits wrong."
- **Session verdict** — a session counts as detected when more than half its scoreable windows are
  flagged, so a detector that fires once and goes quiet on a continuously abnormal recording is not
  credited, and a single bad window doesn't condemn an otherwise normal one.

Metrics are implemented over numpy in `utils/metrics.py` rather than scikit-learn: the ROC and PR curves
step through groups of equal scores, so a detector assigning many windows the same value isn't credited
with an ordering it never produced; AUPRC is the step-wise average precision rather than a trapezoidal
integral, which would credit thresholds that don't exist.

---

## 8. Two failure modes this design avoids

### 8.1 Evidence that depends on the thing being detected

Step anchoring yields far fewer labelled windows than sliding anchors, and the shortfall lands
disproportionately on the abnormal recordings — a recording tested on a handful of windows has not
really been tested, and the reason it yielded few is that the step detector struggled *because* the gait
was abnormal. Sliding anchors give every recording the same coverage per second, whatever its gait looks
like, so evidence yield no longer depends on the thing being measured.

### 8.2 A preprocessing stage that helps anomalies hide

Cross-correlation alignment optimises each window's similarity to a template of normal gait. Applied to
an anomalous window, it searches for the least anomalous-looking shift and shrinks the error the score is
built from — degrading the operating point without destroying the ranking, which is why `xcorr` posts a
respectable AUPRC alongside the worst F1 of the four policies.

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

Every plot is written as both `.png` and `.pdf`. Two diagnostics are worth calling out because they are
easy to misread:

- **Alignment sharpness gain** — mean-signal energy after alignment ÷ before. Above 1 means the windows
  now agree on where the pattern is; the exact value depends heavily on how phase-locked the windows
  started (near 1 under `step` anchoring, into the tens under `slide`), so it is a diagnostic of the
  anchoring, not a quality score in itself.
- **Alignment branch split** — under `mixed`, how often the correlation fallback was needed. It is a
  diagnostic of the step detector rather than a second anomaly score, and it moves substantially with the
  detector's tuning, so it should be read qualitatively (is the fallback rare or common) rather than
  compared precisely across runs.

The `ann-1` recording gets its own section, reporting what the detector did on the unlabelled stretches
that are excluded from every metric.

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

# score from the IMU alone, without the cane-tip distance sensor
uv run main.py ae-anomaly --no-dist
```

The CLI runs one configuration per invocation — the shipped defaults, or whatever flags override them.
Seeded throughout (`--seed`, default 0); the same configuration reproduces bit-identical scores across
repeated runs and separate processes.
