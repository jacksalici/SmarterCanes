# SmartCane Analyzer

Analyzes IMU logs (`t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps`) recorded from a walking cane.

## Architecture

- `main.py` — CLI entry point, dispatches to experiments
- `utils/` — shared code (`io.py`: robust CSV loader → `ImuRecording`, handles serial-glitch rows)
- `exp/` — one module per experiment
- `data/` — raw `rec_*.csv` recordings

## Experiments

- **step-count** (`exp/step_count.py`) — band-pass filters acceleration magnitude (0.5–3.5 Hz) around walking cadence, then peak-picks with an adaptive threshold based on the median height of candidate peaks.

## Commands

```bash
# one file
python main.py step-count data/rec_00004.csv --plot out.png

# every csv in a directory
python main.py step-count data --plot-dir out/
```

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
