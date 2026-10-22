# Robustness scripts

One-off analyses behind the paper revision: seed variance and session-level cross-validation for the
anomaly detector, and nested leave-one-session-out for the step counter. They drive the existing
`exp/` code without changing it. Run them from `Analyzer/`; results go to `out/robustness/`, which is
git-ignored.

| script | what it measures | runtime |
|---|---|---|
| `ae_robust.py {seeds\|cv} {dist\|nodist}` | `seeds`: the fixed split with seeds 0–9. `cv`: the 18 normal sessions in 6 duration-balanced grouped folds × 3 partitions, each fold tested on its held-out normal sessions plus every abnormal session | ~5–15 min each |
| `aggregate.py` | mean ± std tables from the four JSON files, per-class score ratios, the pooled out-of-fold false-alarm rate per session | seconds |
| `step_robust.py` | step accuracy in-sample vs nested LOSO (grid over height σ, refractory, prominence), session-bootstrap CI, signed bias, accuracy on abnormal gait | ~1 min |
| `session_health.py` | per normal session: duration, IMU vs ToF step count, distance-sensor bands | seconds |

```bash
for m in seeds cv; do for d in dist nodist; do
  uv run python scripts/ae_robust.py $m $d
done; done
uv run python scripts/aggregate.py
uv run python scripts/step_robust.py
uv run python scripts/session_health.py
```

The four `ae_robust.py` runs are independent and can run in parallel (`OMP_NUM_THREADS=2` each).
