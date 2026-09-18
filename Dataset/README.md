# SmartCane Dataset

IMU recordings from an instrumented walking cane, and the metadata that labels them. Consumed by
`../Analyzer` — see [Analyzer/README.md](../Analyzer/README.md) for the commands that read this data.

## Layout

- `data/` — every `rec_*.csv` recording, flat, no subfolders
- `split.csv` — `filename,split` table labelling each recording `train`, `test` or `walk`
- `description.csv` — semicolon-separated anomaly type per recording number, for the `ae-anomaly`
  report only; nothing in the pipeline branches on it

## CSV format

A session is stored as one or more 30-second segment files, `rec_XXXXX_segNNN_annY.csv`, where `XXXXX`
is the session index, `NNN` the zero-based segment number, and `Y ∈ {-1, 0, 1}` the stop annotation
(shared by every segment of the session). `t_ms` restarts from 0 in each segment; consecutive segments
of one session are read back-to-back as a single continuous recording by `Analyzer/utils/io.py`'s
`load_session`.

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
neither; `Analyzer/utils/io.py` resolves columns by name and exposes `has_dist` / `has_event`.

## The `annY` annotation

`Y` is a per-session label baked into every segment's filename:

| annotation | meaning |
|---|---|
| `ann0` | normal gait throughout |
| `ann1` | abnormal gait throughout |
| `ann-1` | normal, except at least one marked moment (found via the in-recording `event` column) |
| none | legacy recording, predates the annotation — treated as normal |

## `split.csv`

One row per file in `data/`, assigning it to exactly one of:

- **`train`** (23 files) — normal gait only (`ann0`). What `ae-anomaly` fits its alignment template,
  normalization statistics, model weights and decision threshold on.
- **`test`** (27 files) — labelled, normal and anomalous alike. What `ae-anomaly` scores, frozen, with
  `description.csv` supplying the anomaly type per session for its report. Includes `rec_00017`
  (`ann0`, the held-out normal control session) alongside `rec_00018`–`rec_00028` (`ann1`/`ann-1`,
  the labelled anomalies).
- **`walk`** (32 files) — the rest of the pool: ordinary walking with no role in the `ae-anomaly`
  protocol.

`Analyzer`'s exploratory commands (`step-count`, `step-accuracy`, `step-ae`) take a `--split` flag over
this table, but default to neither a single label nor the whole file: their default, `normal`, is every
`ann0`/legacy recording regardless of label — `train`, `walk`, and the `ann0` files inside `test` —
since ordinary gait analysis has no use for the labelled anomalies `ae-anomaly` exists to score. `all`
opts into the whole pool, anomalies included; a comma-separated subset of `train`, `test`, `walk`
selects by label directly.
