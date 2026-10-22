"""Robustness of the AE anomaly detector: seed repeats and session-level CV.

usage (from Analyzer/): uv run python scripts/ae_robust.py {seeds|cv} {dist|nodist} [OUT.json]

seeds: the fixed train/test split, seeds 0-9.
cv:    the 18 normal sessions in 6 duration-balanced grouped folds, 3 random
       partitions; each fold is tested on its held-out normal sessions plus
       every abnormal session. Writes out/robustness/<mode>_<dist>.json by default.
"""
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ANALYZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYZER))
warnings.filterwarnings("ignore")

from exp.ae_anomaly import load_descriptions, run_ae_anomaly  # noqa: E402
from exp.step_ae import StepAEConfig  # noqa: E402
from utils.io import group_sessions, load_split, paths_for_split  # noqa: E402
from utils.metrics import average_precision, roc_auc, threshold_metrics  # noqa: E402

DATA = ANALYZER.parent / "Dataset" / "data"
SPLIT = ANALYZER.parent / "Dataset" / "split.csv"
DESC = load_descriptions(ANALYZER.parent / "Dataset" / "description.csv")


def summarize(res):
    y, s = res.y_true, res.y_score
    m = threshold_metrics(y, s, res.threshold)
    sess = res.test_windows.session_ids[res.scoreable]
    ratio = s / res.threshold
    per_class = defaultdict(list)
    per_session = {}
    for sid in np.unique(sess):
        k = DESC.get(str(sid), "Normal gait")
        per_class[k].extend(ratio[sess == sid].tolist())
        per_session[str(sid)] = {"cls": k, "n": int((sess == sid).sum()),
                                 "flag_rate": float(np.mean(ratio[sess == sid] > 1))}
    return {
        "auprc": average_precision(y, s), "roc_auc": roc_auc(y, s),
        "f1": m.f1, "acc": m.accuracy, "prec": m.precision, "rec": m.recall,
        "spec": m.specificity, "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
        "n": int(y.size), "prev": float(y.mean()),
        "class_median_ratio": {k: float(np.median(v)) for k, v in per_class.items()},
        "per_session": per_session,
        # out-of-fold raw material for pooled metrics
        "ratios": ratio.tolist(), "y": y.tolist(), "sess": [str(x) for x in sess],
    }


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("seeds", "cv") or sys.argv[2] not in ("dist", "nodist"):
        raise SystemExit(__doc__)
    mode, dist = sys.argv[1], sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ANALYZER / "out" / "robustness" / f"{mode}_{dist}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with_dist = dist == "dist"
    results = []
    if mode == "seeds":
        train = paths_for_split(DATA, SPLIT, {"train"})
        test = paths_for_split(DATA, SPLIT, {"test"})
        for seed in range(10):
            res = run_ae_anomaly(train, test, StepAEConfig(seed=seed, with_dist=with_dist), DESC)
            r = summarize(res); r["seed"] = seed
            results.append(r)
            print(seed, f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}", flush=True)
    else:
        all_paths = paths_for_split(DATA, SPLIT, {"train", "test", "walk"})
        sessions = group_sessions(all_paths)
        anom = [p for sid, ps in sessions.items() if sid in DESC for p in ps]
        normal = {sid: ps for sid, ps in sessions.items() if sid not in DESC}
        sizes = {sid: sum(p.stat().st_size for p in ps) for sid, ps in normal.items()}
        K = 6
        for rep in range(3):
            rng = np.random.default_rng(rep)
            # balanced grouped folds: shuffle, then greedily assign largest-first to lightest fold
            ids = list(normal)
            rng.shuffle(ids)
            ids.sort(key=lambda s: -sizes[s])
            folds, load = [[] for _ in range(K)], np.zeros(K)
            for sid in ids:
                j = int(np.argmin(load + rng.random(K) * 1e-6))
                folds[j].append(sid); load[j] += sizes[sid]
            for f, held in enumerate(folds):
                train = [p for sid in normal if sid not in held for p in normal[sid]]
                test = [p for sid in held for p in normal[sid]] + anom
                res = run_ae_anomaly(train, test, StepAEConfig(seed=rep, with_dist=with_dist), DESC)
                r = summarize(res); r.update(rep=rep, fold=f, held=held)
                results.append(r)
                print(rep, f, held, f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f} spec {r['spec']:.3f}", flush=True)
    Path(out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
