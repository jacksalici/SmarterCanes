"""Reporting for the `ae-anomaly` experiment: CSV, markdown and plots.

Kept apart from `ae_anomaly.py` so the evaluation itself stays free of
formatting concerns, and apart from `main.py` because there is rather more of
it than the other experiments' plotting.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from exp.ae_anomaly import LABEL_ANOMALY, LABEL_DROP, AEAnomalyResult
from utils.io import DIST_ERROR_FLOOR_MM
from utils.metrics import pr_curve, roc_curve, threshold_metrics

_NORMAL_COLOR = "tab:blue"
_ANOMALY_COLOR = "tab:red"


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
    """Contiguous stretches of an unlabelled region that scored above threshold.

    For the `ann-1` recording this is the interesting part: windows excluded
    from every metric because nothing marks them, which the detector fired on
    anyway. Returns (start_s, end_s, peak_score) per run.
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


def _counts_str(counts: dict[str, int]) -> str:
    total = max(1, sum(counts.values()))
    names = {"step": "step", "xcorr": "correlation fallback", "none": "unshifted"}
    return ", ".join(
        f"{counts[k]} {names.get(k, k)} ({counts[k] / total:.0%})"
        for k in ("step", "xcorr", "none")
        if k in counts
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


def _agg(values: list[float], nd: int = 3) -> str:
    """mean, with the spread appended once there is more than one run."""
    if len(values) == 1:
        return _fmt(values[0], nd)
    return f"{np.mean(values):.{nd}f} ± {np.std(values):.{nd}f}"


def write_ablation_markdown(
    ablation: list[tuple[str, list["AEAnomalyResult"]]], chosen: "AEAnomalyResult"
) -> list[str]:
    """Markdown table comparing pipeline variants, plus a reading of it."""
    n_seeds = len(ablation[0][1]) if ablation else 1
    lines = ["## Pipeline ablation", ""]
    lines.append(
        "Each row changes one thing from the row above. All of them share the training "
        "data, the test data, the labels and the false-alarm budget, so the differences "
        "are the pipeline's alone. `windows` is how many labelled test windows the variant "
        "produced at all — the first row is the point about step anchoring, not a detail."
    )
    if n_seeds > 1:
        lines.append("")
        lines.append(
            f"Each row is {n_seeds} runs at different random seeds, reported as mean ± standard "
            "deviation. The spread is the thing to read the gaps against: several of these "
            "variants differ by less than it."
        )
    lines.append("")
    lines.append("| variant | labelled windows | AUPRC | ROC-AUC | F1 | recall | sessions right |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, runs in ablation:
        star = " ←" if "mixed align + conv + dist" in name else ""
        sess = [sum(s.correct for s in r.sessions) for r in runs]
        n_sess = len(runs[0].sessions)
        sess_str = f"{sess[0]}/{n_sess}" if n_seeds == 1 else f"{np.mean(sess):.1f}/{n_sess}"
        lines.append(
            f"| {name}{star} | {int(runs[0].y_true.size)} | "
            f"{_agg([r.auprc for r in runs])} | {_agg([r.roc_auc for r in runs])} | "
            f"{_agg([r.window_metrics.f1 for r in runs])} | "
            f"{_agg([r.window_metrics.recall for r in runs])} | {sess_str} |"
        )
    lines.append("")

    first = ablation[0][1][0] if ablation else None
    if first is not None:
        per_session = sorted(first.sessions, key=lambda s: s.n_labeled)[:3]
        thin = ", ".join(f"rec_{s.session_id} ({s.n_labeled})" for s in per_session)
        lines.append(
            f"The first row is the original pipeline. Its problem is visible in the window "
            f"count: step anchoring produced {int(first.y_true.size)} labelled windows against "
            f"{int(chosen.y_true.size)} for the sliding variant, and the shortfall is "
            f"concentrated on the abnormal recordings — the thinnest were {thin}. A detector "
            f"cannot be said to have been tested on a recording it drew one window from, and "
            f"the reason it drew one is that the step detector failed there, which is to say "
            f"it failed *because* the gait was abnormal."
        )
        lines.append("")
    return lines


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
    axes[0, 1].axvline(result.threshold, color="gray", linestyle="--", linewidth=1.0,
                       label=f"threshold (p{result.config.threshold_pct:g})")
    axes[0, 1].set_xscale("log")
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
    fig.savefig(out_path, dpi=150)
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
    axes[1].axhline(result.threshold, color="gray", linestyle="--", linewidth=1.0,
                    label="threshold")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("reconstruction error")
    axes[1].set_title("Per-session score distribution")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_traces(result: AEAnomalyResult, out_path: Path) -> None:
    """Anomaly score against time for every test session.

    This is where a session-level number stops hiding things: a recording can
    sit just under the threshold throughout, or spike once and subside, and
    those are different failures.
    """
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
    for ax, s in zip(axes, sessions):
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
        # Unlabelled stretches (an ann-1 recording away from its marker) are
        # scored but excluded from every metric - shade them so the plot says
        # so. Spans are merged first: windows overlap, so drawing one per
        # window would stack alpha into a gradient that means nothing.
        for lo, hi in _merge_spans(
            [(t[i] - w_set.window_s / 2, t[i] + w_set.window_s / 2)
             for i in np.flatnonzero(labels == LABEL_DROP)]
        ):
            ax.axvspan(lo, hi, color="gray", alpha=0.15, linewidth=0)
        ax.axhline(result.threshold, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_examples(result: AEAnomalyResult, out_path: Path) -> None:
    """What the model reconstructs well and badly, and which channels give it away."""
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def write_markdown(
    result: AEAnomalyResult,
    out_path: Path,
    plot_names: dict[str, str],
    ablation: list[tuple[str, AEAnomalyResult]] | None = None,
) -> None:
    """A standalone report: the protocol, the numbers, and the per-session verdicts."""
    c = result.config
    m, o = result.window_metrics, result.oracle_metrics
    w_set = result.test_windows
    n_train = int((result.train_split == "train").sum())
    n_val = int((result.train_split == "val").sum())

    hit = [s for s in result.sessions if s.correct]
    miss = [s for s in result.sessions if not s.correct]

    lines: list[str] = []
    a = lines.append

    a("# Autoencoder anomaly detection on cane gait — evaluation report")
    a("")
    a(f"Model `{c.model}`, {w_set.window_s:g} s windows at {w_set.fs:g} Hz "
      f"({w_set.window_len} samples × {len(w_set.channels)} channels: "
      f"{', '.join(w_set.channels)}), `anchor={c.anchor}`, "
      f"`align_mode={c.align_mode}`, latent {c.latent}.")
    a("")

    a("## Protocol")
    a("")
    a(f"- **Train** `data/train` — {len(np.unique(result.train_windows.session_ids))} normal "
      f"session(s), {result.train_windows.n_windows} windows "
      f"({n_train} train / {n_val} validation, split on contiguous time blocks with a "
      f"one-window guard band).")
    a(f"- **Test** `data/test` — {len(result.sessions)} session(s), {w_set.n_windows} windows, "
      f"of which {int(result.scoreable.sum())} carry a label.")
    a("- The alignment template, the normalization statistics, the weights and the decision "
      "threshold are all fitted on training data only and applied frozen to the test set. "
      "No test label enters the pipeline at any point.")
    a(f"- **Threshold** = p{c.threshold_pct:g} of the reconstruction error on held-out normal "
      f"*training* windows = `{result.threshold:.5f}`.")
    a("")
    a("Labels follow the filename annotation: `ann0` normal, `ann1` anomalous throughout. "
      "For the single `ann-1` recording only windows overlapping an `event` marker are "
      "positives — the rest are dropped rather than counted as normal, since nothing "
      "records where the abnormal stretch ends.")
    a("")

    if c.align_mode in ("step", "mixed"):
        a("### Alignment policy")
        a("")
        train_counts = result.train_windows.source_counts()
        test_counts = result.test_windows.source_counts()
        a(f"`align_mode={c.align_mode}` places a window by centering the nearest *detected step*, "
          f"and falls back to correlating the acceleration magnitude against the normal template "
          f"only where no step was detected within ±{c.max_lag_s:g} s. The step branch is preferred "
          f"because it has no bias: it is placement, not matching, and the step detector never sees "
          f"the template. Correlation picks the shift that makes a window look most like normal "
          f"gait, which is a favour done to precisely the windows that should score badly.")
        a("")
        a(f"Branch split — training: {_counts_str(train_counts)}; "
          f"test: {_counts_str(test_counts)}.")
        a("")
        a("The split is not incidental. How often a recording needs the fallback is itself a "
          "readout of how much its gait still looks like stepping:")
        a("")
        a("| session | label | windows | step branch | fallback | fallback rate |")
        a("|---|---|---|---|---|---|")
        w = result.test_windows
        for sess in sorted(result.sessions, key=lambda s: (s.label, s.session_id)):
            mask = w.session_ids == sess.session_id
            src = w.align_source[mask]
            n_step = int(np.sum(src == "step"))
            n_fall = int(np.sum(src == "xcorr"))
            total = max(1, int(mask.sum()))
            a(f"| rec_{sess.session_id} | {_label_name(sess.label)} | {int(mask.sum())} | "
              f"{n_step} | {n_fall} | {n_fall / total:.0%} |")
        a("")

    a("## Window-level results")
    a("")
    a(f"{int(result.y_true.size)} labelled windows, "
      f"{int(result.y_true.sum())} anomalous ({result.prevalence:.1%} prevalence).")
    a("")
    a("| metric | value | note |")
    a("|---|---|---|")
    a(f"| **AUPRC** | **{_fmt(result.auprc)}** | threshold-free; "
      f"{result.prevalence:.3f} is chance at this prevalence |")
    a(f"| **ROC-AUC** | **{_fmt(result.roc_auc)}** | threshold-free; 0.5 is chance |")
    a(f"| **F1** | **{_fmt(m.f1)}** | at the calibrated threshold |")
    a(f"| **Accuracy** | **{_fmt(m.accuracy)}** | at the calibrated threshold |")
    a(f"| Precision | {_fmt(m.precision)} | |")
    a(f"| Recall | {_fmt(m.recall)} | |")
    a(f"| Specificity | {_fmt(m.specificity)} | |")
    a(f"| Balanced accuracy | {_fmt(m.balanced_accuracy)} | |")
    a(f"| F1 at the best possible threshold | {_fmt(o.f1)} | oracle — peeks at test labels, "
      f"not an achievable operating point |")
    a("")
    a(f"Confusion at the calibrated threshold: TP {m.tp}, FP {m.fp}, TN {m.tn}, FN {m.fn}.")
    a("")
    a("The gap between F1 and the oracle F1 is the part of the loss that is threshold "
      "placement rather than ranking: AUPRC and ROC-AUC measure whether the score orders "
      "the windows correctly at all, F1 measures whether the cut lands in the right place.")
    a("")

    a("### Choice of operating point")
    a("")
    a("The threshold is a percentile of the error on held-out normal *training* windows, "
      "which makes it a stated false-alarm budget rather than something tuned on results: "
      "p99 means \"accept a 1% false-alarm rate on normal gait\". The table below is that "
      "budget swept, to show what the choice costs. Every row is reachable without "
      "consulting a test label.")
    a("")
    a("| budget | threshold | precision | recall | F1 | accuracy | specificity |")
    a("|---|---|---|---|---|---|---|")
    for pct in (90.0, 95.0, 97.5, 99.0, 100.0):
        thr = float(np.percentile(result.val_scores, pct))
        tm = threshold_metrics(result.y_true, result.y_score, thr)
        star = " ←" if pct == c.threshold_pct else ""
        a(f"| p{pct:g}{star} | {thr:.5f} | {_fmt(tm.precision)} | {_fmt(tm.recall)} | "
          f"{_fmt(tm.f1)} | {_fmt(tm.accuracy)} | {_fmt(tm.specificity)} |")
    a("")

    a("## Per-session results")
    a("")
    a(f"A session counts as *detected* when more than half its scoreable windows are flagged. "
      f"**{len(hit)}/{len(result.sessions)}** sessions came out right "
      f"({result.session_accuracy:.0%}).")
    a("")
    a("| session | label | description | windows | flagged | flag rate | median error | "
      "max error | verdict |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(result.sessions, key=lambda s: (s.label, s.session_id)):
        a(f"| rec_{s.session_id} | {_label_name(s.label)} | {s.description or '—'} | "
          f"{s.n_labeled} | {s.n_flagged} | {s.flag_rate:.2f} | {s.median_score:.4f} | "
          f"{s.max_score:.4f} | {'✅ correct' if s.correct else '❌ **missed**'} |")
    a("")

    if miss:
        a("### Where it fails")
        a("")
        for s in miss:
            kind = "false negative" if s.label == LABEL_ANOMALY else "false positive"
            a(f"- **rec_{s.session_id}** ({kind}) — {s.description or 'no description'}. "
              f"{s.n_flagged}/{s.n_labeled} windows flagged, median error "
              f"{s.median_score:.4f} against a threshold of {result.threshold:.4f}.")
        a("")
    else:
        a("Every test session came out on the right side.")
        a("")

    if c.with_dist:
        a("## The distance channel")
        a("")
        a("`dist_mm` — the cane tip's height above the ground — is included here as an eighth "
          "input channel. It is normally excluded by default because it is the sensor the step "
          "detector treats as ground truth, so feeding it to a model that is also *anchored* on "
          "detected steps would be circular. That objection is weaker here: with sliding anchors "
          "the step detector no longer decides which windows exist, so the distance reading is "
          "just another sensor.")
        a("")
        a("Readings below the sensor's error floor "
          f"({DIST_ERROR_FLOOR_MM:g} mm) are measurement failures rather than a tip on the "
          "ground, and are clamped to the floor exactly as the step detector clamps them. That "
          "clamping matters, because dropout is not evenly spread:")
        a("")
        a("| session | label | dropout clamped | description |")
        a("|---|---|---|---|")
        for sess in sorted(result.sessions, key=lambda s: (s.label, s.session_id)):
            frac = sess.dist_clipped_frac
            a(f"| rec_{sess.session_id} | {_label_name(sess.label)} | "
              f"{'—' if frac is None else f'{frac:.1%}'} | {sess.description or '—'} |")
        a("")
        a("Two sessions carry real dropout and both are abnormal, so for **rec_00026** and "
          "**rec_00028** part of any gain from this channel could be the model recognising a "
          "sensor losing its return rather than a gait going wrong. The other ten test sessions "
          "and all five training sessions are clean, so the improvement seen on the tremor, "
          "short-step and unsteadiness recordings cannot be explained that way — for those, the "
          "channel is carrying genuine tip-height information.")
        a("")

    partial = [s for s in result.sessions if s.ann == -1]
    if partial:
        a("## The partially-annotated recording")
        a("")
        for sess in partial:
            runs = unlabelled_flagged_runs(result, sess.session_id)
            n_unlabelled = int(
                np.sum(
                    (w_set.session_ids == sess.session_id) & (result.labels == LABEL_DROP)
                )
            )
            a(f"`rec_{sess.session_id}` ({sess.description or 'no description'}) is annotated "
              f"`ann-1`: a recording that is mostly ordinary walking but contains at least one "
              f"abnormal moment, located only by an in-recording `event` click. It yields "
              f"{sess.n_labeled} labelled window(s) around that marker — all "
              f"{sess.n_flagged} of them flagged — and {n_unlabelled} unlabelled windows that "
              f"are scored but excluded from every metric above.")
            a("")
            if runs:
                a(f"Those excluded windows are not quiet. The detector fires on "
                  f"**{len(runs)} separate stretch(es)** of the unannotated part:")
                a("")
                a("| stretch | peak error | vs threshold |")
                a("|---|---|---|")
                for lo, hi, peak in runs:
                    a(f"| {lo:.1f}–{hi:.1f} s | {peak:.3f} | "
                      f"{peak / result.threshold:.1f}× |")
                a("")
                a("This cuts both ways and the recording cannot settle it. If the session "
                  "contains several falls and only one was clicked, these are correct "
                  "detections that the labels cannot credit; if it contains exactly one, they "
                  "are false positives that the metrics above never charged. Either way the "
                  "numbers in this report exclude them, so none of them helped the score — "
                  "and the description of this recording says *falls*, plural.")
            else:
                a("No unlabelled stretch of this recording exceeded the threshold.")
            a("")

    if ablation:
        lines.extend(write_ablation_markdown(ablation, result))

    a("## Plots")
    a("")
    for caption, name in plot_names.items():
        a(f"### {caption}")
        a("")
        a(f"![{caption}]({name})")
        a("")

    a("## Caveats")
    a("")
    a(f"- Training data is {len(np.unique(result.train_windows.session_ids))} session(s) of one "
      "walker. Nothing here shows the model generalizes to a different person or a "
      "differently-mounted cane.")
    a("- Normal test windows come from a single session (rec_00017), so the false-positive "
      "rate is measured on one walk and is correspondingly uncertain.")
    a("- The `ann1` recordings are labelled anomalous end to end, including the seconds of "
      "ordinary walking that inevitably start and finish each one. Some windows counted as "
      "false negatives are windows of genuinely normal gait inside an anomalous recording.")
    a("- Sliding windows overlap, so the labelled windows are not independent samples; "
      "treat the metrics as descriptive of these recordings, not as estimates with "
      "confidence intervals.")
    a("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
