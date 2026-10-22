"""Summarise the JSON results written by ae_robust.py.

usage (from Analyzer/): uv run python scripts/aggregate.py [RESULTS_DIR]  (default out/robustness)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ANALYZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYZER))
from utils.metrics import average_precision, roc_auc, threshold_metrics  # noqa: E402

S = Path(sys.argv[1]) if len(sys.argv) > 1 else ANALYZER / "out" / "robustness"
KEYS = ["auprc", "roc_auc", "f1", "acc", "prec", "rec", "spec"]
ORDER = ["Normal gait", "Short and irregular steps", "Tremors and short steps", "Various falls",
         "Anomalous long steps", "Unsteadiness"]


def ms(v):
    v = np.asarray(v, float)
    return f"{100*np.nanmean(v):5.1f} ± {100*np.nanstd(v):4.1f}"


def table(rs, title):
    print(f"\n### {title}  (n={len(rs)} runs)")
    print("  " + "  ".join(f"{k}: {ms([r[k] for r in rs])}" for k in KEYS))
    print(f"  FN {np.mean([r['fn'] for r in rs]):.1f}  FP {np.mean([r['fp'] for r in rs]):.1f}"
          f"  n_windows {np.mean([r['n'] for r in rs]):.0f}  prevalence {np.mean([r['prev'] for r in rs]):.2f}")
    print("  per-class median score/threshold:")
    for c in ORDER:
        v = [r["class_median_ratio"][c] for r in rs if c in r["class_median_ratio"]]
        print(f"    {c:28s} {np.mean(v):6.2f} ± {np.std(v):5.2f}  [min {np.min(v):.2f}, max {np.max(v):.2f}]")
    # how often is the per-class ordering exactly as in the paper?
    ok = 0
    for r in rs:
        m = [r["class_median_ratio"][c] for c in ORDER]
        ok += all(a < b for a, b in zip(m, m[1:]))
    print(f"  runs with exact paper ordering of classes: {ok}/{len(rs)}")
    sess_ok = [np.mean([(p['flag_rate'] >= .5) == (p['cls'] != 'Normal gait') for p in r['per_session'].values()]) for r in rs]
    print(f"  session-level accuracy: {ms(sess_ok)}")


def pooled_cv(rs, title):
    """Each normal window scored once, by the fold model that never saw its session;
    anomalous windows scored by every fold model, ratio averaged across folds of a repeat."""
    out = []
    for rep in sorted({r["rep"] for r in rs}):
        fr = [r for r in rs if r["rep"] == rep]
        normal = np.concatenate([np.array(r["ratios"])[np.array(r["y"]) == 0] for r in fr])
        normal_sess = np.concatenate([np.array(r["sess"])[np.array(r["y"]) == 0] for r in fr])
        an = [np.array(r["ratios"])[np.array(r["y"]) == 1] for r in fr]
        assert len({a.size for a in an}) == 1
        anom = np.mean(an, axis=0)
        y = np.r_[np.zeros(normal.size), np.ones(anom.size)]
        s = np.r_[normal, anom]
        m = threshold_metrics(y, s, 1.0)
        fpr_sess = {sid: float(np.mean(normal[normal_sess == sid] > 1)) for sid in np.unique(normal_sess)}
        out.append(dict(auprc=average_precision(y, s), roc_auc=roc_auc(y, s), f1=m.f1, acc=m.accuracy,
                        prec=m.precision, rec=m.recall, spec=m.specificity, n_norm=normal.size,
                        n_anom=anom.size, fpr_sess=fpr_sess, fp=m.fp, fn=m.fn))
    print(f"\n### {title} POOLED out-of-fold (n={len(out)} repeats)")
    print("  " + "  ".join(f"{k}: {ms([o[k] for o in out])}" for k in KEYS))
    print(f"  normal windows {out[0]['n_norm']} / anomalous {out[0]['n_anom']}  FP {np.mean([o['fp'] for o in out]):.1f} FN {np.mean([o['fn'] for o in out]):.1f}")
    fs = defaultdict(list)
    for o in out:
        for k, v in o["fpr_sess"].items():
            fs[k].append(v)
    print("  false-alarm rate per held-out normal session: " +
          ", ".join(f"{k[-2:]}:{100*np.mean(v):.0f}%" for k, v in sorted(fs.items())))


for name in ["seeds_dist", "seeds_nodist", "cv_dist", "cv_nodist"]:
    p = S / f"{name}.json"
    if not p.exists():
        print(f"\n({name} not done)"); continue
    rs = json.loads(p.read_text())
    table(rs, name)
    if name.startswith("cv"):
        pooled_cv(rs, name)
