# Results

Measured outcomes for the cane gait anomaly detector. The design and the reasoning behind each choice
are in [EXPERIMENT.md](EXPERIMENT.md); the command reference is in [README.md](README.md).

Every number here comes from the configuration that ships as the default, fitted on `data/train` and
evaluated on `data/test`. Regenerate with:

```bash
uv run main.py ae-anomaly --ablation --ablation-seeds 3
```

The auto-generated equivalent lands in `out/ae_anomaly/REPORT.md`; this document adds the comparisons
that need several runs.

---

## 1. Configuration

| | |
|---|---|
| **Model** | 1-D conv autoencoder, latent 16, ~34k parameters |
| **Input** | 8 channels × 150 samples (6 s at 25 Hz) |
| **Channels** | `ax ay az gx gy gz \|acc\| dist_mm` |
| **Anchoring** | sliding, 1.5 s hop |
| **Alignment** | `mixed` — nearest detected step, cross-correlation fallback |
| **Score** | `chan_norm` — per-channel error ÷ its median on held-out normal windows |
| **Threshold** | p95 of held-out normal *training* error = `1.951` |
| **Train** | 397 windows from 5 normal sessions (303 train / 78 validation / 16 guard) |
| **Test** | 364 windows from 12 sessions; 290 labelled, 142 anomalous (49.0% prevalence) |

Nothing from the test set enters the fit — not the alignment template, the normalization statistics,
the per-channel score scaling, the weights, or the threshold.

---

## 2. Headline

| metric | value | chance |
|---|---|---|
| **AUPRC** | **0.990** | 0.490 (the positive rate) |
| **ROC-AUC** | **0.989** | 0.500 |
| **F1** | **0.957** | — |
| **Accuracy** | **0.959** | — |
| Precision | 0.978 | |
| Recall | 0.937 | |
| Specificity | 0.980 | |
| Balanced accuracy | 0.959 | |
| Oracle F1 *(best any threshold could do)* | 0.964 | |

Confusion at the calibrated threshold: **TP 133 · FP 3 · TN 145 · FN 9**.

**Session-level: 12 / 12 correct (100%).**

The oracle F1 sits only 0.007 above the achieved F1, so the calibrated threshold is very nearly
optimally placed — what remains is a ranking limit, not an operating-point one.

---

## 3. By anomaly type

Anomaly types come from `description.csv`. Plot: `out/ae_anomaly/error_by_type.png`.

| type | recordings | windows | median error | ÷ threshold | flagged | flag rate |
|---|---|---|---|---|---|---|
| **Normal gait** | 1 | 148 | 1.16 | 0.59× | 3 | 2% |
| Tremors and short steps | 4 | 40 | 5.25 | 2.69× | 40 | 100% |
| Short and irregular steps | 4 | 55 | 2.59 | 1.33× | 47 | 85% |
| Anomalous long steps | 1 | 15 | 19.07 | 9.78× | 14 | 93% |
| Unsteadiness | 1 | 29 | 24.00 | 12.30× | 29 | 100% |
| Various falls | 1 | 3 | 6.00 | 3.08× | 3 | 100% |

The ordering is interpretable. **Unsteadiness** (12.3×) and **anomalous long steps** (9.8×) deform
the signal most — the first throws the gyro channels across ±10 standard deviations, the second is
exactly what a tip-height sensor is built to see. **Short and irregular steps** remains the hardest
class at 1.33×: it leaves each individual step looking roughly normal and changes mainly the *timing
between* steps, which a shape-based reconstruction score is least sensitive to. It is still detected —
85% of its windows — but with the smallest margin, and it is where any further work should go.

**Various falls** contributes only 3 labelled windows, because `rec_00028` carries a single event
marker; see §7.

---

## 4. Per-session

A session counts as detected when more than half its scoreable windows are flagged.

| session | type | windows | flagged | flag rate | median error | ÷ threshold | verdict |
|---|---|---|---|---|---|---|---|
| rec_00017 | — *(normal control)* | 148 | 3 | 0.02 | 1.16 | 0.59× | ✅ |
| rec_00018 | Tremors and short steps | 9 | 9 | 1.00 | 9.54 | 4.89× | ✅ |
| rec_00019 | Tremors and short steps | 8 | 8 | 1.00 | 5.31 | 2.72× | ✅ |
| rec_00020 | Tremors and short steps | 9 | 9 | 1.00 | 5.47 | 2.80× | ✅ |
| rec_00021 | Tremors and short steps | 14 | 14 | 1.00 | 3.56 | 1.83× | ✅ |
| rec_00022 | Short and irregular steps | 17 | 12 | 0.71 | 2.23 | 1.14× | ✅ |
| rec_00023 | Short and irregular steps | 16 | 13 | 0.81 | 2.21 | 1.13× | ✅ |
| rec_00024 | Short and irregular steps | 12 | 12 | 1.00 | 3.65 | 1.87× | ✅ |
| rec_00025 | Short and irregular steps | 10 | 10 | 1.00 | 2.74 | 1.40× | ✅ |
| rec_00026 | Anomalous long steps | 15 | 14 | 0.93 | 19.07 | 9.78× | ✅ |
| rec_00027 | Unsteadiness | 29 | 29 | 1.00 | 24.00 | 12.30× | ✅ |
| rec_00028 | Various falls | 3 | 3 | 1.00 | 6.00 | 3.08× | ✅ |

The three false positives are all in `rec_00017`, out of 148 windows.

---

## 5. Ablation

A cumulative path from the original pipeline to the shipped default, one change per row. Three seeds
each, mean ± standard deviation. The last row is asserted in code to equal the shipped defaults.

| # | change | labelled windows | AUPRC | F1 | sessions |
|---|---|---|---|---|---|
| 1 | the original pipeline | **161** | 0.894 ± 0.004 | 0.825 | 10.0 / 12 |
| 2 | + sliding anchors | 292 | 0.926 ± 0.002 | 0.816 | 9.0 / 12 |
| 3 | + conv model (latent 16) | 292 | 0.958 ± 0.004 | 0.835 | 9.3 / 12 |
| 4 | + no alignment | 305 | 0.952 ± 0.001 | 0.843 | 9.3 / 12 |
| 5 | + step alignment | 291 | 0.966 ± 0.004 | 0.847 | 10.0 / 12 |
| 6 | + mixed alignment | 290 | 0.961 ± 0.004 | 0.853 | 10.0 / 12 |
| 7 | + distance channel | 290 | 0.981 ± 0.002 | 0.912 | 11.0 / 12 |
| 8 | **+ per-channel scoring** *(default)* | 290 | **0.991 ± 0.001** | **0.953** | **12.0 / 12** |

Reading it:

- **Row 1's window count is the finding, not a detail.** Step anchoring produced 161 labelled windows
  against 290, and the shortfall lands on the abnormal recordings — `rec_00018`, `rec_00019` and
  `rec_00027` yielded **one window each**. A detector cannot be said to have been tested on a
  recording it drew one window from, and the reason it drew one is that the step detector failed
  there — which is to say it failed *because* the gait was abnormal. Row 1's AUPRC is measured on a
  test set that quietly excluded its hardest cases, so it flatters the original pipeline.
- **Row 2 looks like a regression and is not.** Moving to sliding anchors drops F1 and session
  accuracy because the newly-included windows are the hard ones. The metric got worse; the
  measurement got honest.
- **Rows 4–6 isolate the alignment policy** (§6.1).
- **Rows 7–8 are synergistic, not additive** (§6.2).

---

## 6. Controlled comparisons

### 6.1 Alignment policy

Five seeds, 7 channels, `score_agg=mean`, everything else at defaults.

| policy | placement rule | AUPRC | F1 | sessions |
|---|---|---|---|---|
| `none` | left at the anchor | 0.952 ± 0.004 | 0.846 | 9.2 / 12 |
| `xcorr` | best correlation against the normal template | 0.955 ± 0.005 | **0.831** | 9.2 / 12 |
| `step` | nearest detected step centred | **0.963 ± 0.005** | 0.849 | 10.0 / 12 |
| `mixed` *(default)* | step where detected, correlation as fallback | 0.958 ± 0.005 | **0.850** | 10.0 / 12 |

Both step-based policies beat both alternatives on F1 and session accuracy. The diagnostic detail is
that **`xcorr` posts the worst F1 of the four despite a respectable AUPRC** — it does not wreck the
ranking, it degrades the operating point, which is exactly the signature predicted for a stage that
lets anomalous windows search for the most normal-looking shift.

`step` and `mixed` are separated by less than the seed spread; this table distinguishes step-based
from correlation-based placement, not `step` from `mixed`.

#### The fallback rate is a diagnostic in its own right

Under `mixed`, how often a recording needs the correlation fallback is a readout of how much its gait
still looks like stepping — and it needs no model at all.

| | step branch | fallback | fallback rate |
|---|---|---|---|
| Training data (5 normal sessions) | 360 | 37 | **9%** |
| `rec_00017` (normal control) | 137 | 11 | **7%** |
| Abnormal recordings | 77 | 139 | **43–93%** |

Per recording, the extremes are `rec_00027` (unsteadiness, 93% fallback) and `rec_00025` (90%).

### 6.2 Distance channel × score aggregation

Three seeds, at defaults except the two fields varied.

| channels | aggregation | AUPRC | F1 | sessions |
|---|---|---|---|---|
| 7 (`--no-dist`) | `mean` | 0.958 ± 0.004 | 0.847 | 10.0 / 12 |
| 7 (`--no-dist`) | `chan_norm` | 0.941 | 0.816 | 8 / 12 |
| 8 | `mean` | 0.981 ± 0.002 | 0.912 | 11.0 / 12 |
| 8 | `chan_norm` *(default)* | **0.991 ± 0.001** | **0.953** | **12.0 / 12** |

**These two changes are synergistic, and the interaction has a mechanism.** `chan_norm` divides each
channel's error by its median on held-out normal windows. At seven homogeneous IMU channels that
rescaling buys nothing and in fact *hurts* (0.941 vs 0.958) — it amplifies quiet channels that carry
no signal. Add `dist_mm`, measured in millimetres alongside accelerations in g, and the channels'
natural residuals differ by orders of magnitude: a plain mean becomes a weighted vote dominated by
whichever channel is noisiest. Rescaling then recovers the distance channel's contribution, and the
pair together is worth +0.033 AUPRC and two sessions over either alone.

The biggest single beneficiary is the **short and irregular steps** class, which goes from 56% to 85%
of windows flagged and from two missed sessions to none.

#### One confound in the distance channel, and it is contained

Readings below the sensor's 90 mm error floor are measurement failures, not a tip 9 mm off the
ground, and are clamped to the floor. That dropout is not evenly spread:

| session | label | dropout clamped |
|---|---|---|
| `rec_00026` | anomalous | **11.4%** |
| `rec_00028` | anomalous | **10.2%** |
| `rec_00027` | anomalous | 0.1% |
| all other test sessions, all training sessions | — | 0.0% |

For those two sessions, part of the gain could be the model recognising a failing sensor rather than
a failing gait. The other ten test sessions and all five training sessions are clean, so the
improvements on the tremor, short-step and unsteadiness recordings cannot be explained that way.

### 6.3 Operating point

The threshold is a percentile of held-out normal *training* error, which makes it a declared
false-alarm budget rather than something tuned on results. Every row is reachable without consulting
a test label.

| budget | threshold | precision | recall | F1 | accuracy | specificity |
|---|---|---|---|---|---|---|
| p90 | 1.733 | 0.951 | 0.965 | 0.958 | 0.959 | 0.953 |
| **p95** *(default)* | **1.951** | **0.978** | **0.937** | **0.957** | **0.959** | **0.980** |
| p97.5 | 2.039 | 0.978 | 0.923 | 0.949 | 0.952 | 0.980 |
| p99 | 2.519 | 1.000 | 0.810 | 0.895 | 0.907 | 1.000 |
| p100 | 2.742 | 1.000 | 0.768 | 0.869 | 0.886 | 1.000 |

The curve is now flat between p90 and p97.5 — F1 varies by 0.009 across that range. Earlier
configurations were far more threshold-sensitive, which is itself evidence the score separates the
classes better.

### 6.4 Hyperparameter sensitivity

Three seeds, one field varied at a time around the default.

| | AUPRC | F1 | sessions |
|---|---|---|---|
| **Latent width** | | | |
| 8 | 0.976 ± 0.001 | 0.904 | 10.7 / 12 |
| **16** *(default)* | 0.981 ± 0.002 | 0.912 | 11.0 / 12 |
| 32 | 0.984 ± 0.002 | 0.917 | 11.0 / 12 |
| **Window length** | | | |
| 4 s (hop 1.0 s) | 0.962 ± 0.003 | 0.883 | 11.3 / 12 |
| **6 s (hop 1.5 s)** *(default)* | 0.981 ± 0.002 | 0.912 | 11.0 / 12 |
| 8 s (hop 2.0 s) | 0.985 ± 0.002 | 0.921 | 11.0 / 12 |
| **Resampling rate** | | | |
| **25 Hz** *(default)* | 0.981 ± 0.002 | 0.912 | 11.0 / 12 |
| 50 Hz | 0.981 ± 0.001 | 0.886 | 10.0 / 12 |
| **Score aggregation** | | | |
| `mean` | 0.981 ± 0.002 | 0.912 | 11.0 / 12 |
| **`chan_norm`** *(default)* | **0.991 ± 0.001** | **0.953** | **12.0 / 12** |
| `peak` | 0.975 ± 0.003 | 0.771 | 8.0 / 12 |

*(These runs predate the `chan_norm` default, so the non-aggregation rows use `mean` as their
baseline — they measure the effect of each field, not the absolute headline.)*

Latent 32 and an 8 s window are marginally ahead of the defaults, by 0.003–0.004 AUPRC against a
±0.002 spread — one to two standard deviations, and in the opposite direction to an earlier sweep on
less training data. The defaults sit on a plateau rather than a peak, and chasing those fractions
would be fitting the test set. A 4 s window is worse on AUPRC but best on session count, which is
what a shorter window should do: more windows per recording, each noisier.

`peak` is clearly bad (F1 0.771). Averaging the worst tenth of samples amplifies ordinary strike
transients in normal gait as readily as genuine anomalies.

---

## 7. The partially-annotated recording

`rec_00028` ("Various falls", 124.6 s) is annotated `ann-1`: mostly ordinary walking with at least
one abnormal moment, located only by a single `event` click at 7.5 s. It yields **3 labelled windows**
— all 3 flagged — and 74 unlabelled windows that are scored but excluded from every metric above.

Those excluded windows are not quiet:

| stretch | peak error | vs threshold | annotated? |
|---|---|---|---|
| 8.0 – 21.2 s | 31.53 | 16.2× | yes — the one marker |
| 24.4 – 38.2 s | 14.25 | 7.3× | **no** |
| 41.0 – 58.4 s | 26.56 | 13.6× | **no** |
| 63.9 – 73.1 s | 4.36 | 2.2× | **no** |

The remaining ~50 s stays below threshold. This cuts both ways and the recording cannot settle it: if
the session contains several falls and only one was clicked, these are correct detections the labels
cannot credit; if it contains exactly one, they are false positives the metrics never charged. Either
way none of them influenced any number in this document — and the description says *falls*, plural.

**Actionable:** adding markers for the other stretches would settle it and strengthen the test set.

---

## 8. Plots

All in `out/ae_anomaly/`.

| file | what it shows |
|---|---|
| `error_by_type.png` | reconstruction error, normal in blue against each anomaly type in shades of red — stacked histogram plus a per-window strip plot |
| `overview.png` | loss curves, score separation, ROC, precision-recall |
| `sessions.png` | per-session flag rate and score spread |
| `traces.png` | anomaly score against time for every test session |
| `examples.png` | best and worst reconstructions, with per-channel error |

`out/ae_anomaly_nodist/` holds the same set for the 7-channel configuration.

---

## 9. What these numbers do not establish

Restating the limits from [EXPERIMENT.md](EXPERIMENT.md) §10, because 12/12 invites over-reading:

1. **One walker, one mounting.** Five training sessions from the same person. Nothing here shows
   generalization to another person's gait or a differently-mounted cane. This is the single largest
   caveat.
2. **One normal test session.** Specificity 0.980 is measured on `rec_00017` alone.
3. **Overlapping windows are not independent samples**, so these are descriptive statistics, not
   estimates with confidence intervals. 290 labelled windows come from 12 recordings.
4. **Some anomaly types have one recording.** "Unsteadiness", "anomalous long steps" and "various
   falls" are each a single session — their per-type rows describe that recording, not the class.
5. **Configuration was partly chosen on this test set.** The comparisons in §6 all use the same test
   data as the headline, so 0.990 is mildly optimistic. The spread across the grid — AUPRC 0.94–0.99
   — is the more honest range.
6. **Whole-recording labels are coarse.** `ann1` recordings include the seconds of ordinary walking
   that start and end each one, visible as the ramp at the start of several traces; some windows
   counted as false negatives are genuinely normal gait.
