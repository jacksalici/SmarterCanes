"""Per normal session: duration, IMU vs ToF step counts, and distance-sensor health.

usage (from Analyzer/): uv run python scripts/session_health.py
"""
import sys
from pathlib import Path
ANALYZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYZER))
import numpy as np, warnings; warnings.filterwarnings('ignore')
from utils.io import group_sessions, load_session, DIST_ERROR_FLOOR_MM, DIST_STEP_THRESHOLD_MM
from exp.step_count import count_steps
ps = sorted((ANALYZER.parent / 'Dataset' / 'data').glob('*_ann0.csv'))
for sid, p in sorted(group_sessions(ps).items()):
    r = load_session(p); c = count_steps(r); dur = r.t[-1]
    d = r.dist_mm
    print(sid, f"dur {dur:5.0f}s  imu {c.n_steps:3d} tof {c.gt_n_steps:3d}  cad_imu {c.n_steps/dur*60:4.1f}  "
          f"dist<floor {np.mean(d<DIST_ERROR_FLOOR_MM):.0%}  dist>lift {np.mean(d>DIST_STEP_THRESHOLD_MM):.0%}  median {np.median(d):.0f}mm")
