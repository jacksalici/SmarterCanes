# SmartCane Analyzer

Analyzes IMU logs recorded from an instrumented walking cane.

## Architecture

- `main.py` — CLI entry point, dispatches to experiments; owns all printing and plotting
- `utils/` — shared code
  - `io.py` — robust CSV loader → `ImuRecording` (name-based columns, tolerates serial-glitch rows)
  - `windows.py` — windowing preprocessing: resampling, step-anchored windowing, cross-correlation alignment, normalization
- `exp/` — one module per experiment
- `data/` — raw `rec_*.csv` recordings
- `out/` — generated plots and model checkpoints (git-ignored)

## Experiments

- **step-count** (`exp/step_count.py`) — counts steps as sharp isolated excursions of the acceleration
  magnitude away from 1 g, with the height threshold set from each recording's own noise floor
  (mean + k·std of `|acc|-1g`). A cane strike is a broadband mechanical shock rather than a smooth
  oscillation, so the earlier 0.5–3.5 Hz cadence band-pass smeared one impulse into a ringing filter
  response and over-counted by roughly 3×. Cross-checked against a ground-truth count derived from the
  distance sensor with a three-band hysteresis latch (`dist_mm`: 110–120 mm resting, >125 mm tip lifted,
  <110 mm sensor error).

- **step-ae** (`exp/step_ae.py`) — a small MLP autoencoder over aligned multi-step windows, trained on
  normal data only, using reconstruction error as an anomaly score. Where step-count answers *how many*
  steps, this asks what a step *looks like*. See below.

## step-ae pipeline

Four preprocessing stages (`utils/windows.py`), then the model (`exp/step_ae.py`).

1. **Resample.** The firmware's sample interval jitters — the median rate across `data/` is ~67 Hz
   against a nominal 100 Hz — so a window length in seconds would otherwise map to a different number
   of samples in every recording. Every channel is linearly interpolated onto a uniform grid at
   `--target-fs` (default 25 Hz, enough for gait shape and the strike transient while keeping the input
   dimension small). Step and event indices are mapped by nearest sample; event markers are impulses,
   so interpolating them would smear or erase them.

2. **Window.** One window per detected step, taken from `count_steps`, centered on that step, of
   `--window-s` seconds (default 6.0 ≈ three steps: the median interval between detected steps in
   `data/` is 1.94 s, cane-assisted gait here being slow and deliberate — a plain walking cadence would
   put three steps nearer 3 s). `--step-hop` anchors on every Nth step instead. A window is kept only if
   it fits inside its recording with `--max-lag-s` of slack on both sides, so the alignment stage can
   shift it by any lag in range and still re-slice real samples — never zero padding, and never a window
   whose length depends on how far it moved.

3. **Align.** Step anchoring gets windows roughly phase-locked; the residual jitter is removed by
   normalized cross-correlation against a common template, so the same point of the gait cycle lands at
   the same index in every window and the model doesn't spend capacity modelling phase as if it were
   signal. The alignment signal is mean-removed `|acc|` — orientation-independent, since the cane is not
   held at a repeatable angle, and where the tip strike is sharpest. The template is refined iteratively
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
   alignment to before — averaging misaligned copies of one pattern cancels it out, so above 1 means the
   windows now agree on where the pattern is (1.4× on the current `data/`).

4. **Normalize.** Per-channel z-score, with statistics computed over the *training* windows only, and
   global rather than per-window: normalizing each window on its own would erase amplitude differences,
   which for anomaly detection are exactly the thing worth noticing.

**Channels.** `ax, ay, az, gx, gy, gz, |acc|`. `dist_mm` is added as an eighth channel only under
`--with-dist`: it is the ground-truth sensor for step detection, so including it by default would leak
the signal the model should be independent of.

**Model.** A symmetric MLP autoencoder over the flattened `(channels × samples)` window — encoder
widths from `--hidden` (default `128,32`) down to a `--latent` bottleneck (default 8), mirrored back,
ReLU between and a linear output (the target is z-scored, so squashing it would clip exactly the large
excursions that carry the anomaly signal). Adam, MSE, early stopping on validation loss.

Trained on normal windows only, so a bottleneck far narrower than the input forces it to learn the few
degrees of freedom ordinary gait actually has. Reconstruction error is then the anomaly score, and the
decision threshold is a high percentile (`--threshold-pct`, default 99) of the error on *held-out
normal* windows — calibrated on what normal looks like rather than on any assumption about anomalies,
which matters because the labels are not settled yet.

The train/validation split is **by session**, not by window: consecutive windows overlap heavily, so a
window-level split would put near-duplicates on both sides and make the validation loss meaningless.
The template and the normalization statistics are likewise fitted on normal training windows and then
applied frozen, so nothing about the held-out windows leaks into the preprocessing. A checkpoint stores
the weights alongside the template and statistics, so `--load` scores new recordings through exactly the
pipeline the model was trained on.

**Labels.** Which recordings are anomalous is still to be specified, so two label sources are wired up
behind `--normal-by`:

- `event` (default) — a window containing an `event == 1` sample (a single click during recording, a
  user-flagged moment of interest) is held out as a candidate anomaly. Recordings with no `event`
  column are all normal, so older data stays usable.
- `ann` — sessions whose filename annotation is in `--normal-ann` (default `-1`) are normal, the rest
  are candidates. `data/` currently holds only `ann-1` and `ann1`, no `ann0`, and what the values mean
  is provisional.

**Caveat on scale.** `data/` currently yields ~430 windows, fewer than the input dimension (7 × 150).
The bottleneck, `--weight-decay` and the by-session validation split are all there to keep that honest;
the train/validation gap in the loss curves is the thing to check before trusting a score.

## Commands

```bash
# step counting: one session
uv run main.py step-count data/rec_00004_seg000_ann1.csv --plot out.png

# step counting: every recording in a directory, grouped into sessions
uv run main.py step-count data --plot-dir out/

# autoencoder: train on all recordings and score them
uv run main.py step-ae data --plot-dir out/step_ae

# a different window length, and windows anchored on every other step
uv run main.py step-ae data --window-s 3.5 --step-hop 2 --plot-dir out/step_ae

# use the filename annotation as the label source instead of the event marker
uv run main.py step-ae data --normal-by ann --normal-ann -1

# score with a saved model instead of training (reuses its template and statistics)
uv run main.py step-ae data --load out/step_ae/model.pt --plot-dir out/step_ae
```

The `step-ae` plot has five panels: the alignment signal's mean ± std before and after alignment (does
alignment sharpen the common pattern?), the lag histogram (lags piling up at the bounds means
`--max-lag-s` is too tight), the loss curves, the reconstruction-error distribution against the
threshold, and the worst-reconstructed window against its reconstruction.

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
