# Results

Measured outcomes for the cane gait anomaly detector. The design and its rationale are in
[EXPERIMENT.md](EXPERIMENT.md); the command reference is in [README.md](README.md).

**Provenance.** Every number below was produced by the code in this repository at its current state,
seed 0, on `data/train` (5 normal sessions) and `data/test` (12 labelled sessions). Runs are
deterministic: the same configuration reproduces bit-identical scores across repeated runs and
separate processes.

> **Results depend on the step-detector tuning.** `exp/step_count.py` carries three hand-calibrated
> constants, currently `_STEP_MIN_INTERVAL_S = 1.15`, `_STEP_HEIGHT_SIGMA_K = 3.2`,
> `_MIN_HEIGHT_G = 0.30`. These set step times, which set step-anchored windows and the `mixed`
> alignment branch split. Everything here was measured under that tuning; an earlier, stricter tuning
> gave materially different numbers for the step-anchored rows in particular (§5).

Reproduce the shipped configuration (§1–§4, §7) with:

```bash
uv run main.py ae-anomaly
```

The multi-variant tables (§5, §6, §8) came from a variant runner that has since been removed: the CLI
now runs one configuration per invocation and generates only CSVs and plots, never prose. Reproducing
those tables means calling `run_ae_anomaly` directly with the overrides and seed range each table
names.

---

## 1. Configuration

| | |
|---|---|
| Model | 1-D convolutional autoencoder, latent 16, ~34k parameters |
| Input | 8 channels × 150 samples (6 s windows at 25 Hz) |
| Channels | `ax ay az gx gy gz \|acc\| dist_mm` |
| Anchoring | sliding, 1.5 s hop |
| Alignment | `mixed` — nearest detected step centred, cross-correlation fallback |
| Scoring | `chan_norm` — per-channel error scaled by its held-out-normal median |
| Threshold | p95 of held-out normal *training* error = **2.063** |

These are the shipped defaults; `uv run main.py ae-anomaly` with no flags reproduces §2.

**Dataset.** 397 training windows from 5 sessions (303 train / 78 validation / 16 discarded as guard
band). 364 test windows from 12 sessions, of which **290 carry a label** — 142 anomalous, 148 normal,
49.0% prevalence. The 74 unlabelled windows are the unmarked stretches of the `ann-1` recording
(§7).

---

## 2. Headline result

| metric | value | note |
|---|---|---|
| **AUPRC** | **0.993** | chance = 0.490 (the positive rate) |
| **ROC-AUC** | **0.991** | chance = 0.5 |
| **F1** | **0.942** | at the calibrated threshold |
| **Accuracy** | **0.945** | |
| Precision | 0.977 | |
| Recall | 0.908 | |
| Specificity | 0.980 | |
| Oracle F1 | 0.972 | best threshold in hindsight — not achievable, see below |
| **Sessions correct** | **12 / 12** | majority of scoreable windows flagged |

Confusion at the calibrated threshold: **TP 129, FP 3, TN 145, FN 13**.

The small gap between F1 (0.942) and oracle F1 (0.972) says the remaining loss is mostly *ranking*,
not threshold placement — the score orders windows almost perfectly and the p95 cut sits close to
where it should.

---

## 3. Per anomaly type

Types come from `description.csv`. "÷ thr" is the type's median error divided by the threshold, so
1.00 sits exactly on the decision boundary.

| type | recordings | windows | median error | ÷ thr | flagged |
|---|---|---|---|---|---|
| Normal gait | 1 | 148 | 1.18 | 0.57 | 3 / 148 |
| Tremors and short steps | 4 | 40 | 5.28 | 2.56 | 40 / 40 |
| Short and irregular steps | 4 | 55 | 2.72 | 1.32 | 44 / 55 |
| Anomalous long steps | 1 | 15 | 19.85 | 9.62 | 14 / 15 |
| Unsteadiness | 1 | 29 | 25.20 | 12.21 | 29 / 29 |
| Various falls | 1 | 3 | 6.15 | 2.98 | 2 / 3 |

![Reconstruction error by anomaly type](out/ae_anomaly/error_by_type.png)

The separation is ordered and interpretable. **Unsteadiness** (12.2×) and **long steps** (9.6×) are
caught by an order of magnitude — both deform the waveform or the tip trajectory violently.
**Tremors** (2.6×) and **falls** (3.0×) are comfortably clear.

**Short and irregular steps remains the hard class** at 1.32× and 44/55 windows flagged. It is the
one type whose distribution still overlaps normal gait, which is coherent rather than noisy: a
reconstruction-error score measures how unfamiliar a *waveform* is, and short irregular steps leave
each individual step looking roughly normal while changing mainly the *timing between* steps.
Catching that class by a wide margin would need an explicitly rhythmic feature, not a better
autoencoder. Per-channel scoring (§6) narrowed the gap considerably — it was 0.9–1.2× and two missed
sessions under plain mean scoring — but did not close it.

---

## 4. Per session

| session | label | type | windows | flagged | flag rate | median ÷ thr | verdict |
|---|---|---|---|---|---|---|---|
| rec_00017 | normal | — | 148 | 3 | 0.02 | 0.57 | ✅ |
| rec_00018 | anomalous | Tremors and short steps | 9 | 9 | 1.00 | 4.62 | ✅ |
| rec_00019 | anomalous | Tremors and short steps | 8 | 8 | 1.00 | 2.56 | ✅ |
| rec_00020 | anomalous | Tremors and short steps | 9 | 9 | 1.00 | 2.64 | ✅ |
| rec_00021 | anomalous | Tremors and short steps | 14 | 14 | 1.00 | 1.77 | ✅ |
| rec_00022 | anomalous | Short and irregular steps | 17 | 11 | 0.65 | 1.19 | ✅ |
| rec_00023 | anomalous | Short and irregular steps | 16 | 11 | 0.69 | 1.16 | ✅ |
| rec_00024 | anomalous | Short and irregular steps | 12 | 12 | 1.00 | 1.72 | ✅ |
| rec_00025 | anomalous | Short and irregular steps | 10 | 10 | 1.00 | 1.31 | ✅ |
| rec_00026 | anomalous | Anomalous long steps | 15 | 14 | 0.93 | 9.62 | ✅ |
| rec_00027 | anomalous | Unsteadiness | 29 | 29 | 1.00 | 12.21 | ✅ |
| rec_00028 | anomalous | Various falls | 3 | 2 | 0.67 | 2.98 | ✅ |

All twelve sessions land on the right side. The three false positives are isolated windows inside
rec_00017 — 2% of a 148-window normal walk — not a stretch the detector believes is abnormal.

### Choice of operating point

The threshold is a percentile of held-out normal *training* error, so moving it is choosing a
false-alarm budget rather than fitting to the answer. Every row is reachable without a test label.

| budget | threshold | precision | recall | F1 | accuracy | specificity |
|---|---|---|---|---|---|---|
| p90 (10% false alarms) | 1.672 | 0.945 | 0.972 | **0.958** | 0.959 | 0.946 |
| **p95 (5%)** ← default | 2.063 | 0.977 | 0.908 | 0.942 | 0.945 | 0.980 |
| p97.5 | 2.161 | 0.977 | 0.894 | 0.934 | 0.938 | 0.980 |
| p99 (1%) | 2.638 | 1.000 | 0.810 | 0.895 | 0.907 | 1.000 |

p90 buys the best F1; p99 buys perfect precision. p95 is the shipped compromise.

---

## 5. Ablation

A cumulative path from the original pipeline to the current default, each row adding exactly one
change. Three seeds per row, mean ± standard deviation. Row 8 is the shipped configuration, and its
AUPRC/F1/session figures agree with §2 as they must.

Measured with the variant runner that has since been removed from the CLI (§Provenance), so this
table cannot be regenerated by `uv run main.py ae-anomaly` alone.

| # | pipeline | labelled windows | AUPRC | F1 | sessions |
|---|---|---|---|---|---|
| 1 | the original pipeline | 196 | 0.944 ± 0.002 | 0.858 | 10.0 / 12 |
| 2 | + sliding anchors | 292 | 0.926 ± 0.002 | 0.816 | 9.0 / 12 |
| 3 | + conv model (latent 16) | 292 | 0.958 ± 0.004 | 0.835 | 9.3 / 12 |
| 4 | + no alignment | 305 | 0.952 ± 0.001 | 0.843 | 9.3 / 12 |
| 5 | + step alignment | 291 | 0.959 ± 0.005 | 0.847 | 10.0 / 12 |
| 6 | + mixed alignment | 290 | 0.956 ± 0.003 | 0.843 | 9.7 / 12 |
| 7 | + distance channel | 290 | 0.980 ± 0.000 | 0.907 | 11.0 / 12 |
| 8 | **+ per-channel scoring (default)** | 290 | **0.992 ± 0.001** | **0.947** | **12.0 / 12** |

Read with care — this table does not say every change was an improvement:

- **Row 2 looks like a regression, and on these metrics it is.** Sliding anchors cost AUPRC and a
  session. But rows 1 and 2 are not scored on the same test set: row 1 labels 196 windows, row 2
  labels 292. The extra 96 windows are disproportionately the abnormal recordings the step detector
  under-samples, so row 2 is being asked a harder question. The case for sliding anchors is that the
  question is the right one, not that it improves the answer (EXPERIMENT §8.1).
- **Rows 4–6 are close.** Step (0.959), mixed (0.956) and no alignment (0.952) sit within about one
  standard deviation of each other, and `mixed` is not the best of them in isolation. What separates
  them is a bias argument rather than these numbers: cross-correlation against a *normal* template
  lets an abnormal window search for the least abnormal-looking shift (EXPERIMENT §4.3).
- **Rows 7 and 8 are the two unambiguous wins**, both far outside seed noise: +0.024 AUPRC for the
  distance channel and +0.012 more for per-channel scoring, together worth two sessions.

---

## 6. The two changes that mattered

### 6.1 The distance channel

`dist_mm` (cane-tip height) as an eighth input, measured against the same pipeline without it:

| | AUPRC | ROC-AUC | F1 | accuracy | recall | sessions |
|---|---|---|---|---|---|---|
| 7 channels (`--no-dist`) | 0.941 | 0.922 | 0.798 | 0.834 | 0.669 | 8 / 12 |
| **8 channels (default)** | **0.993** | **0.991** | **0.942** | **0.945** | **0.908** | **12 / 12** |

Almost all of the gain is **recall**: 0.669 → 0.908, i.e. 47 false negatives down to 13. Precision is
essentially unchanged. The channel is not making the detector bolder, it is making abnormal gait
visible that the IMU alone could not distinguish.

**Caveat.** Readings below the 90 mm sensor error floor are clamped, and that dropout is not evenly
spread — 11.4% of rec_00026 and 10.2% of rec_00028, ~0% everywhere else, both abnormal recordings.
For those two sessions part of the gain could be the model recognising a failing sensor rather than a
failing gait. The other ten test sessions are clean, so the gains on tremors, short steps and
unsteadiness cannot be explained that way.

### 6.2 Per-channel score scaling

| `--score-agg` | AUPRC | F1 | accuracy | recall | sessions |
|---|---|---|---|---|---|
| `mean` | 0.980 | 0.904 | 0.914 | 0.831 | 10 / 12 |
| **`chan_norm`** (default) | **0.993** | **0.942** | **0.945** | **0.908** | **12 / 12** |

Dividing each channel's error by its held-out-normal median before averaging. This matters much more
at eight channels than seven: `dist_mm` is in millimetres and the accelerometers in g, so their
natural residuals differ by orders of magnitude and a plain mean is effectively a weighted vote
dominated by whichever channel is noisiest. At seven channels the two aggregations are a tie.

The third option, `peak` (mean of the worst tenth of samples), was clearly worse — F1 0.771,
8/12 sessions — so brief-event sensitivity is not what this task needs. (That pair of numbers predates
the step-detector retune; the ordering is not in doubt, the exact values are.)

---

## 7. The partially-annotated recording

`rec_00028` ("various falls", 124.6 s) is `ann-1`: mostly ordinary walking with one `event` marker at
7.5 s. It contributes **3 labelled windows** (2 flagged) and **74 unlabelled windows** excluded from
every metric above.

Those excluded windows are not quiet — the detector fires on several separate stretches of the
unannotated part, separated by long quiet passages. The recording cannot settle what they are: if it
contains several falls and only one was clicked, they are correct detections the labels cannot
credit; if it contains exactly one, they are false positives the metrics never charged. None of them
helped the score either way. The description says *falls*, plural — **adding markers for the
remaining events would make this recording usable evidence instead of ambiguous**.

---

## 8. Hyperparameter sensitivity

One parameter varied at a time around the default, 3 seeds each. Measured before the step-detector
retune, so treat these as relative comparisons rather than absolute values.

| | AUPRC | F1 | sessions |
|---|---|---|---|
| latent 8 | 0.976 ± 0.001 | 0.904 | 10.7 / 12 |
| **latent 16** ← default | 0.981 ± 0.002 | 0.912 | 11.0 / 12 |
| latent 32 | 0.984 ± 0.002 | 0.917 | 11.0 / 12 |
| window 4 s | 0.962 ± 0.003 | 0.883 | 11.3 / 12 |
| **window 6 s** ← default | 0.981 ± 0.002 | 0.912 | 11.0 / 12 |
| window 8 s | 0.985 ± 0.002 | 0.921 | 11.0 / 12 |
| **25 Hz** ← default | 0.981 ± 0.002 | 0.912 | 11.0 / 12 |
| 50 Hz | 0.981 ± 0.001 | 0.886 | 10.0 / 12 |

The defaults sit on a plateau rather than a peak. Latent 32 and an 8 s window are marginally ahead on
AUPRC (~2 SD) while a 4 s window gives the best session count — no setting dominates, and the spread
across the whole grid (0.962–0.985) is the honest measure of how much this choice is worth. Doubling
the sample rate buys nothing and costs F1.