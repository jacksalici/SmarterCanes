"""Fig. 2: one row per class, every test window a dot, in units of the threshold, from the session CV.

Each dot is a window's out-of-fold score / fold threshold, averaged over the models that never saw
its session: the 3 that held out a normal session, all 18 for an abnormal one. Rows are ordered by
median; the black tick is the class median of those dots, the grey bar the min-max range of the
per-fold class median, and the solid line the p95 threshold, at 1.

usage (from Analyzer/): uv run python scripts/fig2_strip.py [CV_JSON]
  default: out/robustness/cv_dist.json (from scripts/ae_robust.py cv dist)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

ANALYZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYZER))
from exp.ae_anomaly import load_descriptions  # noqa: E402

NORMAL = "Normal gait"
SHORT_NAMES = {
    NORMAL: "Normal gait",
    "Short and irregular steps": "Short, irregular steps",
    "Tremors and short steps": "Tremors, short steps",
    "Various falls": "Falls",
    "Anomalous long steps": "Long steps",
    "Unsteadiness": "Unsteadiness",
}


def main():
    cv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ANALYZER / "out" / "robustness" / "cv_dist.json"
    desc = load_descriptions(ANALYZER.parent / "Dataset" / "description.csv")
    folds = json.loads(cv_path.read_text())

    # out-of-fold scores per window, keyed by (session, index of the window within its session)
    per_window = defaultdict(list)
    per_fold_median = defaultdict(list)
    for fold in folds:
        seen = defaultdict(int)
        for sid, ratio in zip(fold["sess"], fold["ratios"]):
            sid = sid.removeprefix("rec_")
            per_window[(sid, seen[sid])].append(ratio)
            seen[sid] += 1
        for cls, v in fold["class_median_ratio"].items():
            per_fold_median[cls].append(v)
    cv_range = {cls: (min(v), max(v)) for cls, v in per_fold_median.items()}

    ratios, sessions = defaultdict(list), defaultdict(set)
    for (sid, _), v in sorted(per_window.items()):
        cls = desc.get(sid, NORMAL)
        expected = len(folds) if cls != NORMAL else len({f["rep"] for f in folds})
        assert len(v) == expected, (sid, len(v), expected)
        ratios[cls].append(float(np.mean(v)))
        sessions[cls].add(sid)
    for cls in sorted(ratios, key=lambda c: np.median(ratios[c])):
        print(f"{cls:28s} windows {len(ratios[cls]):4d}  sessions {len(sessions[cls]):2d}  "
              f"median {np.median(ratios[cls]):.2f}  fold-median range {cv_range[cls][0]:.2f}-{cv_range[cls][1]:.2f}  "
              f"flagged {np.mean(np.array(ratios[cls]) > 1):.0%}")

    classes = sorted(ratios, key=lambda c: np.median(ratios[c]))
    rng = np.random.default_rng(0)

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 1.65))
    for row, cls in enumerate(classes):
        y = len(classes) - 1 - row
        v = np.array(ratios[cls])
        if cls in cv_range:
            lo, hi = cv_range[cls]
            ax.plot([lo, hi], [y, y], color="0.75", linewidth=6, solid_capstyle="butt", zorder=1)
        color = "tab:blue" if cls == NORMAL else "tab:red"
        ax.scatter(v, y + rng.uniform(-0.25, 0.25, v.size), s=11, color=color, alpha=0.5,
                   linewidths=0, zorder=2)
        ax.plot([np.median(v)] * 2, [y - 0.38, y + 0.38], color="black", linewidth=1.4, zorder=3)

    ax.axvline(1.0, color="black", linewidth=1.0, zorder=0)
    ax.text(1.06, len(classes) - 0.45, "threshold", fontsize=7, va="bottom", ha="left")

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([
        f"{SHORT_NAMES.get(c, c)} ({len(ratios[c])}/{len(sessions[c])})" for c in reversed(classes)
    ])
    ax.set_xscale("log")
    ticks = [0.3, 1, 3, 10, 30]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.25, 40)
    ax.set_ylim(-0.55, len(classes) - 0.3)
    ax.set_xlabel("anomaly score / threshold (log)")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fig.tight_layout(pad=0.3)
    out = cv_path.parent / "fig2_strip"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"written {out}.pdf/.png")


if __name__ == "__main__":
    main()
