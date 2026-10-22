"""Step counting: in-sample vs nested leave-one-session-out, bias, and abnormal gait.

usage (from Analyzer/): uv run python scripts/step_robust.py
"""
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ANALYZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYZER))
import exp.step_count as sc  # noqa: E402
from utils.io import load_csv  # noqa: E402

DATA = ANALYZER.parent / "Dataset" / "data"


def detect(rec, k, refr, prom):
    dev = np.abs(rec.acc_mag - 1.0)
    h = max(dev.mean() + k * dev.std(), sc._MIN_HEIGHT_G)
    p, _ = find_peaks(dev, height=h, distance=max(1, int(refr * rec.fs)), prominence=prom * h)
    return p.size


def windows(pattern, min_gt=5, min_cad=20.0):
    out = []
    for path in sorted(DATA.glob(pattern)):
        rec = load_csv(path)
        if not rec.has_dist:
            continue
        gt = sc.count_steps(rec).gt_n_steps or 0
        dur = float(rec.t[-1])
        if gt < min_gt or gt / dur * 60 < min_cad:
            continue
        sid = re.search(r"rec_(\d+)_", path.name).group(1)
        out.append((sid, path.name, rec, gt))
    return out


def acc(det, gt):
    return max(0.0, 1 - abs(det - gt) / gt)


GRID = list(itertools.product(np.arange(2.4, 4.01, 0.2), np.arange(0.8, 1.51, 0.05), [0.3, 0.4, 0.5, 0.6, 0.7]))


def main():
    ws = windows("*_ann0.csv")
    sids = sorted({w[0] for w in ws})
    print(f"normal windows {len(ws)} from {len(sids)} sessions:",
          {s: sum(w[0] == s for w in ws) for s in sids})
    # cache detections over the grid
    D = np.array([[detect(w[2], *g) for g in GRID] for w in ws])
    G = np.array([w[3] for w in ws])
    A = np.clip(1 - np.abs(D - G[:, None]) / G[:, None], 0, None)
    S = np.array([w[0] for w in ws])

    shipped = GRID.index(min(GRID, key=lambda g: abs(g[0] - 3.2) + abs(g[1] - 1.15) + abs(g[2] - 0.5)))
    a = A[:, shipped]
    print(f"shipped params {GRID[shipped]}: in-sample {a.mean():.4f} ± {a.std():.4f}")
    best = int(A.mean(0).argmax())
    print(f"grid optimum {GRID[best]}: {A[:, best].mean():.4f}")

    # nested LOSO: choose params on other sessions, score held-out session
    oof = np.zeros(len(ws)); chosen = []
    for s in sids:
        tr, te = S != s, S == s
        g = int(A[tr].mean(0).argmax())
        oof[te] = A[te, g]; chosen.append(GRID[g])
    print(f"nested LOSO: {oof.mean():.4f} ± {oof.std():.4f}  (chosen: {sorted(set((round(c[0],1),round(c[1],2),c[2]) for c in chosen))})")
    per_sess = [oof[S == s].mean() for s in sids]
    print(f"  per-session mean of means {np.mean(per_sess):.4f}, min {np.min(per_sess):.4f}")

    det = D[:, shipped]
    print(f"signed bias (det-gt)/gt: {np.mean((det-G)/G):+.4f}; total detected {det.sum()} vs gt {G.sum()} "
          f"({(det.sum()-G.sum())/G.sum():+.2%}); exact windows {np.mean(det==G):.0%}; "
          f"over {np.sum(det>G)}, under {np.sum(det<G)}")
    # bootstrap over sessions
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(5000):
        pick = rng.choice(sids, len(sids))
        bs.append(np.concatenate([oof[S == p] for p in pick]).mean())
    print(f"  nested LOSO 95% session-bootstrap CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]")

    # abnormal gait, shipped params, not tuned on it
    for pat, name in [("*_ann1.csv", "ann1"), ("*_ann-1.csv", "ann-1")]:
        aw = windows(pat)
        allw = windows(pat, 0, 0)
        if not aw:
            print(f"{name}: 0 scorable windows (of {len(allw)})"); continue
        aa = np.array([acc(detect(w[2], 3.2, 1.15, 0.5), w[3]) for w in aw])
        dd = np.array([detect(w[2], 3.2, 1.15, 0.5) for w in aw]); gg = np.array([w[3] for w in aw])
        print(f"{name}: {len(aw)} scorable windows of {len(allw)}, acc {aa.mean():.4f} ± {aa.std():.4f}, "
              f"bias {np.mean((dd-gg)/gg):+.3f}")
        for w, x, d in zip(aw, aa, dd):
            print(f"   {w[1]} gt {w[3]} det {d} acc {x:.2f}")


if __name__ == "__main__":
    main()
