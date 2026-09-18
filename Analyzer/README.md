# SmartCane Analyzer

Analyzes IMU logs recorded from an instrumented walking cane.

## Architecture

- `main.py` — CLI entry point, dispatches to experiments; owns all printing and plotting
- `utils/` — shared code
  - `io.py` — robust CSV loader → `ImuRecording` (name-based columns, tolerates serial-glitch rows)
  - `windows.py` — windowing preprocessing: resampling, step-anchored or sliding windowing, four alignment policies, normalization
  - `metrics.py` — ROC, precision-recall, AUPRC and threshold metrics over numpy
- `exp/` — one module per experiment (`ae_report.py` is the reporting half of `ae_anomaly.py`)
- `out/` — generated plots and model checkpoints (git-ignored)

The recordings live outside the Analyzer, in `../Dataset/` — see
[Dataset/README.md](../Dataset/README.md) for the file layout, the CSV format and the `split.csv` /
`description.csv` metadata. `--split` (on `step-count`, `step-accuracy`, `step-ae`) selects recordings
from `Dataset/data` by their `Dataset/split.csv` label, defaulting to `normal` (every `ann0`/legacy
recording); `ae-anomaly` always fits on `train` and scores `test`.

For the full experimental design and rationale, see [EXPERIMENT.md](EXPERIMENT.md).

## Experiments

- **step-count** (`exp/step_count.py`) — counts steps as sharp isolated excursions of the acceleration
  magnitude away from 1 g, with the height threshold set from each recording's own noise floor
  (mean + k·std of `|acc|-1g`). A strike must also clear a refractory period, stand out from its
  surroundings by half the detection height, and exceed an absolute floor. Cross-checked against a
  ground-truth count derived from the distance sensor with a three-band hysteresis latch (`dist_mm`:
  90–110 mm resting, >118 mm tip lifted, <90 mm sensor error).

- **step-accuracy** (`exp/step_accuracy.py`) — scores the detected count against that ground truth, per
  30-second window. Windows are dropped when the ground truth cannot support a comparison: fewer than
  `--min-gt-steps` steps, or a ground-truth cadence below `--min-gt-cadence`. Mean accuracy is 0.952 over
  the 45 scoreable windows, 0.934 under nested leave-one-session-out cross-validation.

- **step-ae** (`exp/step_ae.py`) — an autoencoder over aligned multi-step windows, trained on normal
  data only, using reconstruction error as an anomaly score. Where step-count answers *how many* steps,
  this asks what a step *looks like*. See below.

- **ae-anomaly** (`exp/ae_anomaly.py`, reporting in `exp/ae_report.py`) — the same autoencoder put
  through a train/test protocol: everything is fitted on the `train` split (normal gait only) and
  applied frozen to the `test` split, which carries labels the model never sees. Reports AUPRC,
  ROC-AUC, F1 and accuracy per window plus a per-session verdict. See below.

## step-ae pipeline

Four preprocessing stages (`utils/windows.py`), then the model (`exp/step_ae.py`).

1. **Resample.** The firmware's sample interval jitters — median rate ~67 Hz against a nominal 100 Hz —
   so every channel is linearly interpolated onto a uniform grid at `--target-fs` (default 25 Hz). Step
   and event indices are mapped by nearest sample rather than interpolated, since they are impulses.

2. **Window.** Windows of `--window-s` seconds (default 6.0, about three steps at this dataset's cadence),
   cut every `--hop-s` seconds under `--anchor slide` (default). `--anchor step` instead centres one
   window on each detected step, phase-locking for free but tying evidence yield to a step detector that
   degrades on abnormal gait — see **ae-anomaly** below. A window is kept only if it fits inside its
   recording with `--max-lag-s` of slack on both sides, so alignment can shift it without zero-padding.

3. **Align.** Each window is placed into a common phase by one of four policies (`--align-mode`, see the
   table under **ae-anomaly**); the default `mixed` centres the nearest detected step and falls back to
   cross-correlation only where no step was found. The alignment signal is mean-removed `|acc|` —
   orientation-independent, and where the tip strike is sharpest. For `xcorr` the template is refined
   iteratively (`--align-iters`, default 3) until the mean lag change is under one sample.

4. **Normalize.** Per-channel z-score, with statistics computed over the training windows only and
   global rather than per-window, so amplitude differences (which anomaly detection depends on) survive.

**Channels.** `ax, ay, az, gx, gy, gz, |acc|`, plus `dist_mm` — the cane tip's height above the ground —
as an eighth channel by default. `--no-dist` scores from the IMU alone. The channel is switched off
automatically for any input where some recording lacks the column (older firmware).

**Model.** Either of two autoencoders, chosen with `--model`. `conv` (default) is a 1-D convolutional
autoencoder — strided convolutions down to a `--latent` bottleneck (default 16), mirrored back by
interpolation — about 34k parameters, shift-tolerant. `mlp` is a symmetric MLP autoencoder over the
flattened window (widths from `--hidden`, default `128,32`) — about 279k parameters. Both use ReLU
internally and a linear output (the target is z-scored, so squashing would clip the excursions that
carry the anomaly signal). Adam, MSE, early stopping on validation loss.

Trained on normal windows only; reconstruction error is the anomaly score, and the decision threshold is
a high percentile (`--threshold-pct`, default 95) of the error on held-out normal windows. A model
loaded with `--load` keeps the threshold stored in its checkpoint.

The train/validation split is by session, not by window, since consecutive windows overlap heavily. The
alignment template and normalization statistics are likewise fitted on normal training windows and
applied frozen. A checkpoint stores the weights alongside the template and statistics, so `--load`
scores new recordings through exactly the pipeline the model was trained on.

**Labels.** Two label sources behind `--normal-by`:

- `event` — a window containing an `event == 1` sample (a user click during recording) is held out as a
  candidate anomaly. Recordings with no `event` column are all normal.
- `ann` — the filename annotation:

  | annotation | meaning | treatment |
  |---|---|---|
  | `ann0` | normal throughout | every window is training data |
  | `ann1` | abnormal throughout | every window is a candidate |
  | `ann-1` | normal *except* at least one marked moment | normal per window, unless it contains an `event` marker |
  | none | legacy file | normal |

**Caveat on scale.** The dataset currently yields ~760 step-anchored windows, fewer than the input
dimension (7 × 150 = 1050). The bottleneck, `--weight-decay` and the by-session validation split keep
that honest; the train/validation gap in the loss curves is the thing to check before trusting a score.

## Commands

All four commands default to `../Dataset/data`. `--split` (on `step-count`, `step-accuracy`, `step-ae`)
narrows that directory: `normal` (default) is every `ann0`/legacy recording; `all` is the whole pool,
anomalies included; a comma-separated subset of `train`, `test`, `walk` selects by `Dataset/split.csv`
label.

```bash
# step counting: one session
uv run main.py step-count ../Dataset/data/rec_00004_seg000_ann1.csv --plot out.png

# step counting: every recording in a directory, grouped into sessions
uv run main.py step-count ../Dataset/data --plot-dir out/

# step-count accuracy against the dist_mm ground truth, per window
uv run main.py step-accuracy ../Dataset/data --plot-dir out/step_accuracy

# autoencoder: train on all recordings and score them
uv run main.py step-ae ../Dataset/data --plot-dir out/step_ae

# score with a saved model instead of training
uv run main.py step-ae ../Dataset/data --load out/step_ae/model.pt --plot-dir out/step_ae

# the headline ae-anomaly run - bare defaults are the best-measured configuration
uv run main.py ae-anomaly

# score from the IMU alone, without the cane-tip distance sensor
uv run main.py ae-anomaly --no-dist
```

## ae-anomaly: evaluating the detector

`ae-anomaly` fits the alignment template, the normalization statistics, the weights and the decision
threshold on the `train` split and applies all four frozen to the `test` split. No test label enters the
pipeline at any point.

**Labels** come from the filename annotation. `ann0` is normal and every window is a negative; `ann1` is
abnormal gait throughout and every window is a positive; `ann-1` is a normal recording containing at
least one abnormal moment located only by the in-recording `event` click, so only windows overlapping a
marker are positives and the rest are dropped. `description.csv` supplies the anomaly type per
recording, for reporting only.

**Metrics** are in `utils/metrics.py` (numpy, no scikit-learn): AUPRC and ROC-AUC are threshold-free and
say whether the score orders windows correctly; F1 and accuracy at the calibrated threshold say whether
the cut lands in the right place. AUPRC's chance level is the positive rate, not 0.5.

**Anchoring** defaults to `--anchor slide` rather than one window per detected step, so the amount of
evidence gathered from a recording doesn't depend on a step detector that itself degrades on abnormal
gait. `--align-mode` offers four policies:

| mode | how a window is placed | bias |
|---|---|---|
| `none` | left at its anchor | none, but phase is unhandled |
| `xcorr` | best correlation against the normal template | searches for the most normal-looking shift |
| `step` | nearest detected step centred; unshifted if there is none | none — placement, not matching |
| `mixed` | `step` where a step was detected, `xcorr` only as fallback | fallback only where no step exists |

`mixed` (default) prefers step placement, which has no bias since the step detector never sees the
template, and falls back to correlation only where no step exists.

The distance channel (`dist_mm`) is on by default and helps consistently across alignment policies;
readings below the sensor error floor (`DIST_ERROR_FLOOR_MM`, 90 mm) are clamped rather than treated as
ground contact. `--no-dist` scores from the IMU alone, and the channel disables itself automatically for
recordings that predate the sensor.

The decision threshold is a percentile of the reconstruction error on held-out normal training windows —
a declared false-alarm budget (`--threshold-pct 95` means "accept a 5% false-alarm rate on normal
gait") rather than something tuned on results.

### Output

`--out-dir` (default `out/ae_anomaly/`) receives CSVs and plots.

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

Every plot is written twice, as `.png` and as `.pdf`.

## CSV Format

See [Dataset/README.md](../Dataset/README.md) for the recording filename convention, the CSV column
layout, and the `annY` annotation. `utils/io.py` resolves columns by name and exposes `has_dist` /
`has_event`, so a recording missing either column still loads.

Setup: `uv sync`, then run commands with `uv run`.
