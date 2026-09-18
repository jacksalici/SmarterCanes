"""Outputs for the `ae-anomaly` experiment: CSV tables and plots.

Every figure is written as both PNG and PDF. No generated prose - RESULTS.md
is the only place a claim is made about what these numbers mean.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from exp.ae_anomaly import LABEL_ANOMALY, LABEL_DROP, LABEL_NORMAL, AEAnomalyResult
from utils.io import DIST_ERROR_FLOOR_MM
from utils.metrics import pr_curve, roc_curve, threshold_metrics

def _save(fig, out_path: Path, dpi: int = 150) -> None:
    """Write a figure as both PNG and PDF, whichever suffix was asked for."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_path.with_suffix(".pdf"))


_NORMAL_COLOR = "tab:blue"
_ANOMALY_COLOR = "tab:red"

# Reds for the anomaly types, spread over hue and lightness (not a single-hue
# ramp) so neighbouring types stay distinguishable under transparency.
_ANOMALY_SHADES = (
    "#FFA05C",  # apricot
    "#F2542D",  # vermilion
    "#D01C1F",  # red
    "#9B1B5E",  # crimson-magenta
    "#5A0E33",  # dark wine
)


def _anomaly_palette(n: int) -> list[str]:
    """`n` distinguishable reds, falling back to a colormap if there are many."""
    if n <= len(_ANOMALY_SHADES):
        return list(_ANOMALY_SHADES[:n])
    import matplotlib.pyplot as plt

    return [plt.get_cmap("Reds")(v) for v in np.linspace(0.35, 0.95, n)]


def _log_ticks(ax, axis: str = "x") -> None:
    """Label a log axis with plain numbers at 1-2-5 steps, plus minor ticks -
    denser than matplotlib's default one-label-per-decade.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20))
    target.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    target.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=100))
    target.set_minor_formatter(NullFormatter())
    ax.tick_params(axis=axis, which="major", length=5, labelsize=11)
    ax.tick_params(axis=axis, which="minor", length=2.5)


def _threshold_divider(
    ax, threshold: float, axis: str = "x", fontsize: float = 11, inside: bool = False
) -> None:
    """Draw the decision threshold, labelled "anomaly"/"not anomaly" on each
    side - solid and heavier than any other line, so it doesn't read as one
    more dashed median. `inside` puts the labels below the top axis edge,
    for panels whose top already carries a title.
    """
    label_kw = dict(fontsize=fontsize, fontweight="semibold", zorder=7)
    if axis == "x":
        ax.axvline(threshold, color="black", linewidth=2.2, zorder=6)
        trans = ax.get_xaxis_transform()  # x in data units, y in axes fraction
        y, va = (0.97, "top") if inside else (1.015, "bottom")
        ax.text(threshold, y, "not anomaly  ", transform=trans, ha="right", va=va,
                color=_NORMAL_COLOR, **label_kw)
        ax.text(threshold, y, "  anomaly", transform=trans, ha="left", va=va,
                color=_ANOMALY_COLOR, **label_kw)
    else:
        ax.axhline(threshold, color="black", linewidth=2.2, zorder=6)
        trans = ax.get_yaxis_transform()  # y in data units, x in axes fraction
        x, ha = (0.995, "right") if inside else (1.005, "left")
        ax.text(x, threshold, "anomaly ", transform=trans, ha=ha, va="bottom",
                color=_ANOMALY_COLOR, **label_kw)
        ax.text(x, threshold, "not anomaly ", transform=trans, ha=ha, va="top",
                color=_NORMAL_COLOR, **label_kw)


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Collapse overlapping intervals into the fewest covering the same range."""
    merged: list[tuple[float, float]] = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def unlabelled_flagged_runs(
    result: AEAnomalyResult, session_id: str
) -> list[tuple[float, float, float]]:
    """Contiguous stretches of an unlabelled (`ann-1`) region that scored
    above threshold. Returns (start_s, end_s, peak_score) per run.
    """
    w_set = result.test_windows
    mask = (w_set.session_ids == session_id) & (result.labels == LABEL_DROP)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    idx = idx[np.argsort(w_set.center_times[idx])]

    runs: list[tuple[float, float, float]] = []
    current: list[int] = []
    for i in idx:
        if result.scores[i] > result.threshold:
            current.append(i)
        elif current:
            runs.append(_run_bounds(w_set, result, current))
            current = []
    if current:
        runs.append(_run_bounds(w_set, result, current))
    return runs


def _run_bounds(w_set, result, run: list[int]) -> tuple[float, float, float]:
    half = w_set.window_s / 2
    times = w_set.center_times[run]
    return (
        float(times.min() - half),
        float(times.max() + half),
        float(result.scores[run].max()),
    )


def _label_name(label: int) -> str:
    return {LABEL_ANOMALY: "anomalous", 0: "normal", LABEL_DROP: "unlabelled"}[label]


def write_session_csv(result: AEAnomalyResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "session", "ann", "label", "description", "n_windows", "n_labeled",
                "n_flagged", "flag_rate", "median_score", "max_score",
                "detected", "correct",
            ]
        )
        for s in result.sessions:
            w.writerow(
                [
                    f"rec_{s.session_id}", s.ann, _label_name(s.label), s.description,
                    s.n_windows, s.n_labeled, s.n_flagged, f"{s.flag_rate:.4f}",
                    f"{s.median_score:.5f}", f"{s.max_score:.5f}",
                    "yes" if s.detected else "no", "yes" if s.correct else "no",
                ]
            )


def write_window_csv(result: AEAnomalyResult, out_path: Path) -> None:
    """Every scoreable test window, so a session can be re-examined by hand."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_set = result.test_windows
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "center_time_s", "label", "score", "flagged"])
        for i in range(w_set.n_windows):
            if result.labels[i] == LABEL_DROP:
                continue
            w.writerow(
                [
                    f"rec_{w_set.session_ids[i]}", f"{w_set.center_times[i]:.2f}",
                    _label_name(int(result.labels[i])), f"{result.scores[i]:.6f}",
                    "yes" if result.scores[i] > result.threshold else "no",
                ]
            )


def plot_overview(result: AEAnomalyResult, out_path: Path) -> None:
    """Training curves, score separation, and the two threshold-free curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Loss curves - the train/val gap is the overfitting check.
    epochs = np.arange(1, result.train_loss.size + 1)
    axes[0, 0].plot(epochs, result.train_loss, linewidth=0.9, label="train")
    axes[0, 0].plot(epochs, result.val_loss, linewidth=0.9, label="validation")
    axes[0, 0].axvline(result.best_epoch + 1, color="gray", linestyle="--", linewidth=0.7)
    axes[0, 0].set_yscale("log")
    _log_ticks(axes[0, 0], "y")
    axes[0, 0].set_title(
        f"Reconstruction loss (best epoch {result.best_epoch + 1}/{result.train_loss.size})"
    )
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend()

    # 2. Score separation. Log x: the anomalous tail runs orders of magnitude out.
    y, s = result.y_true, result.y_score
    normal, anomalous = s[y == 0], s[y == 1]
    lo = max(min(s.min(), result.val_scores.min()), 1e-6)
    bins = np.geomspace(lo, s.max(), 50)
    axes[0, 1].hist(normal, bins=bins, color=_NORMAL_COLOR, alpha=0.75,
                    label=f"normal test ({normal.size})")
    axes[0, 1].hist(anomalous, bins=bins, color=_ANOMALY_COLOR, alpha=0.65,
                    label=f"anomalous test ({anomalous.size})")
    axes[0, 1].hist(result.val_scores, bins=bins, histtype="step", color="black",
                    linewidth=1.0, label=f"held-out normal train ({result.val_scores.size})")
    axes[0, 1].set_xscale("log")
    _log_ticks(axes[0, 1], "x")
    _threshold_divider(axes[0, 1], result.threshold, "x", fontsize=9, inside=True)
    axes[0, 1].set_title("Reconstruction error per window")
    axes[0, 1].set_xlabel("mean squared error")
    axes[0, 1].set_ylabel("windows")
    axes[0, 1].legend(fontsize=8)

    # 3. ROC.
    fpr, tpr = roc_curve(y, s)
    axes[1, 0].plot(fpr, tpr, linewidth=1.4, color=_ANOMALY_COLOR)
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray", linewidth=0.7, label="chance")
    m = result.window_metrics
    axes[1, 0].plot(1 - m.specificity, m.recall, "ko", markersize=6,
                    label="calibrated threshold")
    axes[1, 0].set_title(f"ROC (AUC = {result.roc_auc:.3f})")
    axes[1, 0].set_xlabel("false positive rate")
    axes[1, 0].set_ylabel("true positive rate")
    axes[1, 0].legend(fontsize=8)

    # 4. Precision-recall. The baseline is prevalence, not 0.5.
    recall, precision = pr_curve(y, s)
    axes[1, 1].plot(recall, precision, linewidth=1.4, color=_ANOMALY_COLOR)
    axes[1, 1].axhline(result.prevalence, linestyle="--", color="gray", linewidth=0.7,
                       label=f"chance = prevalence ({result.prevalence:.2f})")
    axes[1, 1].plot(m.recall, m.precision, "ko", markersize=6, label="calibrated threshold")
    axes[1, 1].set_title(f"Precision-recall (AUPRC = {result.auprc:.3f})")
    axes[1, 1].set_xlabel("recall")
    axes[1, 1].set_ylabel("precision")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def plot_sessions(result: AEAnomalyResult, out_path: Path) -> None:
    """Per-session outcome: how much of each recording was flagged, and how hard."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sessions = sorted(result.sessions, key=lambda s: (s.label, s.session_id))
    names = [f"rec_{s.session_id}" for s in sessions]
    colors = [_ANOMALY_COLOR if s.label == LABEL_ANOMALY else _NORMAL_COLOR for s in sessions]
    x = np.arange(len(sessions))

    fig, axes = plt.subplots(2, 1, figsize=(max(9, 0.75 * len(sessions)), 9), sharex=True)

    axes[0].bar(x, [s.flag_rate for s in sessions], color=colors)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
                    label="session verdict cut (majority)")
    for i, s in enumerate(sessions):
        axes[0].text(i, s.flag_rate + 0.02, "OK" if s.correct else "MISS",
                     ha="center", fontsize=7,
                     color="green" if s.correct else "darkred")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("fraction of windows flagged")
    axes[0].set_title("Per-session flag rate (red = anomalous, blue = normal)")
    axes[0].legend(fontsize=8, loc="upper left")

    # Score spread per session against the threshold, on a log scale.
    data = []
    for s in sessions:
        mask = (result.test_windows.session_ids == s.session_id) & (result.labels != LABEL_DROP)
        data.append(result.scores[mask])
    bp = axes[1].boxplot(data, positions=x, widths=0.6, patch_artist=True, showfliers=True,
                         flierprops=dict(markersize=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
    axes[1].set_yscale("log")
    _log_ticks(axes[1], "y")
    _threshold_divider(axes[1], result.threshold, "y", fontsize=9, inside=True)
    axes[1].set_ylabel("reconstruction error")
    axes[1].set_title("Per-session score distribution")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    # No legend here: the threshold was its only entry and now names itself.

    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def plot_traces(result: AEAnomalyResult, out_path: Path) -> None:
    """Anomaly score against time for every test session."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sessions = sorted(result.sessions, key=lambda s: (s.label, s.session_id))
    n = len(sessions)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    w_set = result.test_windows
    for y_index, (ax, s) in enumerate(zip(axes, sessions)):
        mask = w_set.session_ids == s.session_id
        t = w_set.center_times[mask]
        sc = result.scores[mask]
        labels = result.labels[mask]
        order = np.argsort(t)
        t, sc, labels = t[order], sc[order], labels[order]

        color = _ANOMALY_COLOR if s.label == LABEL_ANOMALY else _NORMAL_COLOR
        ax.plot(t, sc, linewidth=0.9, color=color)
        flagged = sc > result.threshold
        ax.plot(t[flagged], sc[flagged], "o", markersize=2.5, color="black")
        # Shade unlabelled (scored but excluded) stretches; spans merged
        # first since overlapping windows would otherwise stack alpha.
        for lo, hi in _merge_spans(
            [(t[i] - w_set.window_s / 2, t[i] + w_set.window_s / 2)
             for i in np.flatnonzero(labels == LABEL_DROP)]
        ):
            ax.axvspan(lo, hi, color="gray", alpha=0.15, linewidth=0)
        ax.set_yscale("log")
        _log_ticks(ax, "y")
        ax.tick_params(axis="y", which="major", labelsize=7)
        if y_index == 0:  # named once, not on every panel
            _threshold_divider(ax, result.threshold, "y", fontsize=7, inside=True)
        else:
            ax.axhline(result.threshold, color="black", linewidth=1.4, zorder=6)
        verdict = "OK" if s.correct else "MISS"
        ax.set_title(
            f"rec_{s.session_id} ({_label_name(s.label)}) - {verdict}"
            + (f"\n{s.description}" if s.description else ""),
            fontsize=9,
        )
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("error", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Anomaly score over time (black dots = flagged, grey = unlabelled and excluded)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_path)
    plt.close(fig)


def group_scores_by_type(
    result: AEAnomalyResult,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray, int]]]:
    """Split scoreable test windows into normal, then one group per anomaly
    type (from `description.csv`, pooling sessions of the same type).

    Returns (normal_scores, [(anomaly type, scores, n_sessions), ...]).
    """
    w = result.test_windows
    scoreable = result.labels != LABEL_DROP
    normal = result.scores[scoreable & (result.labels == LABEL_NORMAL)]

    groups: dict[str, list[np.ndarray]] = {}
    sessions: dict[str, int] = {}
    order: list[str] = []
    for sess in sorted(result.sessions, key=lambda s: s.session_id):
        if sess.label != LABEL_ANOMALY:
            continue
        name = sess.description or "unspecified"
        if name not in groups:
            groups[name] = []
            sessions[name] = 0
            order.append(name)
        mask = (w.session_ids == sess.session_id) & scoreable
        groups[name].append(result.scores[mask])
        sessions[name] += 1

    return normal, [(name, np.concatenate(groups[name]), sessions[name]) for name in order]


def plot_error_histogram(result: AEAnomalyResult, out_path: Path) -> None:
    """Reconstruction error by anomaly type: normal in blue, each type its own
    red, overlaid with transparency (not stacked) plus a solid outline and a
    dashed median line per type. Kept spare - no title, no y axis - so it can
    drop into a paper or slide with its own caption.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal, groups = group_scores_by_type(result)
    shades = _anomaly_palette(len(groups))

    all_scores = np.concatenate([normal] + [g for _, g, _ in groups])
    bins = np.geomspace(max(all_scores.min(), 1e-6), all_scores.max(), 42)

    fig, ax = plt.subplots(figsize=(12, 3.6))

    series = [("Normal gait", normal, _NORMAL_COLOR)] + [
        (name, g, shades[i]) for i, (name, g, _) in enumerate(groups)
    ]
    for name, scores, color in series:
        ax.hist(scores, bins=bins, color=color, alpha=0.45, zorder=2)
        ax.hist(
            scores, bins=bins, histtype="step", color=color, linewidth=1.6, zorder=3,
            label=name,
        )
        ax.axvline(
            np.median(scores), color=color, linestyle=(0, (4, 2)), linewidth=2.0, zorder=4
        )

    ax.set_xscale("log")
    _log_ticks(ax, "x")
    ax.set_xlabel("reconstruction error", fontsize=12)

    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)  # headroom for the threshold labels
    _threshold_divider(ax, result.threshold, "x", fontsize=12)

    ax.tick_params(axis="y", which="both", labelleft=False, length=0)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.92, borderpad=0.6,
              labelspacing=0.4, handlelength=1.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    _save(fig, out_path, dpi=200)
    plt.close(fig)


def plot_examples(result: AEAnomalyResult, out_path: Path) -> None:
    """Best/worst-reconstructed normal and anomalous windows, with per-channel error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w_set = result.test_windows
    scoreable = np.flatnonzero(result.labels != LABEL_DROP)
    normal_idx = scoreable[result.labels[scoreable] == 0]
    anomaly_idx = scoreable[result.labels[scoreable] == LABEL_ANOMALY]

    picks = []
    if normal_idx.size:
        picks.append(("best-reconstructed normal", normal_idx[np.argmin(result.scores[normal_idx])]))
        picks.append(("worst-reconstructed normal (false positive)",
                      normal_idx[np.argmax(result.scores[normal_idx])]))
    if anomaly_idx.size:
        picks.append(("worst-reconstructed anomalous (clearest catch)",
                      anomaly_idx[np.argmax(result.scores[anomaly_idx])]))
        picks.append(("best-reconstructed anomalous (the miss)",
                      anomaly_idx[np.argmin(result.scores[anomaly_idx])]))

    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 2.8 * len(picks)),
                             gridspec_kw={"width_ratios": [3, 1]})
    axes = np.atleast_2d(axes)
    t_win = np.arange(w_set.window_len) / w_set.fs

    for row, (title, i) in enumerate(picks):
        for c in range(len(w_set.channels)):
            axes[row, 0].plot(t_win, w_set.x[i, c], linewidth=0.7, color="tab:blue", alpha=0.8)
            axes[row, 0].plot(t_win, result.recon[i, c], "--", linewidth=0.7, color=_ANOMALY_COLOR)
        axes[row, 0].plot([], [], color="tab:blue", label="input (all channels)")
        axes[row, 0].plot([], [], "--", color=_ANOMALY_COLOR, label="reconstruction")
        axes[row, 0].set_title(
            f"{title}: rec_{w_set.session_ids[i]} at {w_set.center_times[i]:.1f} s "
            f"(error {result.scores[i]:.4f})",
            fontsize=9,
        )
        axes[row, 0].set_ylabel("normalized")
        axes[row, 0].legend(fontsize=7, loc="upper right")

        axes[row, 1].barh(np.arange(len(w_set.channels)), result.channel_error[i],
                          color="tab:gray")
        axes[row, 1].set_yticks(np.arange(len(w_set.channels)))
        axes[row, 1].set_yticklabels(w_set.channels, fontsize=7)
        axes[row, 1].set_title("error by channel", fontsize=9)
        axes[row, 1].tick_params(labelsize=7)

    axes[-1, 0].set_xlabel("time within window (s)")
    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def write_metrics_csv(result: AEAnomalyResult, out_path: Path) -> None:
    """The headline numbers and the operating-point sweep, as one flat table.

    `headline` rows are single values at the calibrated threshold; `operating
    point` rows sweep the false-alarm budget.
    """
    m, o = result.window_metrics, result.oracle_metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "key", "value"])
        for k, v in [
            ("n_labelled_windows", int(result.y_true.size)),
            ("prevalence", f"{result.prevalence:.4f}"),
            ("threshold", f"{result.threshold:.6f}"),
            ("auprc", f"{result.auprc:.4f}"),
            ("roc_auc", f"{result.roc_auc:.4f}"),
            ("f1", f"{m.f1:.4f}"),
            ("accuracy", f"{m.accuracy:.4f}"),
            ("precision", f"{m.precision:.4f}"),
            ("recall", f"{m.recall:.4f}"),
            ("specificity", f"{m.specificity:.4f}"),
            ("balanced_accuracy", f"{m.balanced_accuracy:.4f}"),
            ("oracle_f1", f"{o.f1:.4f}"),
            ("tp", int(m.tp)), ("fp", int(m.fp)), ("tn", int(m.tn)), ("fn", int(m.fn)),
            ("sessions_correct", sum(s.correct for s in result.sessions)),
            ("sessions_total", len(result.sessions)),
        ]:
            w.writerow(["headline", k, v])

        w.writerow([])
        w.writerow(["operating point", "budget", "threshold", "precision", "recall",
                    "f1", "accuracy", "specificity", "is_default"])
        for pct in (90.0, 95.0, 97.5, 99.0, 100.0):
            thr = float(np.percentile(result.val_scores, pct))
            tm = threshold_metrics(result.y_true, result.y_score, thr)
            w.writerow(["operating point", f"p{pct:g}", f"{thr:.6f}",
                        f"{tm.precision:.4f}", f"{tm.recall:.4f}", f"{tm.f1:.4f}",
                        f"{tm.accuracy:.4f}", f"{tm.specificity:.4f}",
                        "yes" if pct == result.config.threshold_pct else ""])


def write_unlabelled_runs_csv(result: AEAnomalyResult, out_path: Path) -> None:
    """Flagged stretches inside an `ann-1` recording's unlabelled span -
    scored but excluded from every metric, since nothing marks where the
    abnormal stretch ends.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "start_s", "end_s", "peak_error", "times_threshold"])
        for sess in result.sessions:
            if sess.ann != -1:
                continue
            for start, end, peak in unlabelled_flagged_runs(result, sess.session_id):
                w.writerow([f"rec_{sess.session_id}", f"{start:.1f}", f"{end:.1f}",
                            f"{peak:.3f}", f"{peak / result.threshold:.1f}"])
