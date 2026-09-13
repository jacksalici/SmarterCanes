#!/usr/bin/env python3
"""SmartCane IMU analyzer entry point.

Usage:
    python main.py step-count data/rec_00004_seg000_ann1.csv [--plot out.png]
    python main.py step-count data [--plot-dir out/]
    python main.py step-ae data [--window-s 6.0] [--plot-dir out/step_ae]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exp.step_ae import (
    DEFAULT_HOP_S,
    DEFAULT_MAX_LAG_S,
    DEFAULT_TARGET_FS,
    DEFAULT_WINDOW_S,
    StepAEConfig,
    run_step_ae,
)
from exp.ae_anomaly import load_descriptions, run_ae_anomaly
from exp.ae_report import (
    plot_error_histogram,
    plot_examples,
    plot_overview,
    plot_sessions,
    plot_traces,
    write_metrics_csv,
    write_session_csv,
    write_unlabelled_runs_csv,
    write_window_csv,
)
from exp.step_accuracy import compute_window_accuracies, write_csv
from exp.step_count import (
    DIST_REST_HIGH_MM,
    DIST_STEP_THRESHOLD_MM,
    count_steps,
)
from utils.io import group_sessions, load_csv, load_session

# Defaults live on StepAEConfig so the two commands cannot drift apart.
_D = StepAEConfig()


def _save_figure(fig, out_path: Path | str) -> None:
    """Write a figure as both PNG and PDF - raster to look at, vector to print."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))


def _sessions_in(path: Path) -> dict[str, list[Path]]:
    """Resolve a file-or-directory argument into sessions of segment paths."""
    if not path.is_dir():
        return {path.stem: [path]}

    csv_paths = sorted(path.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"no .csv files found in {path}")

    sessions = group_sessions(csv_paths)
    if not sessions:
        raise SystemExit(f"no rec_*.csv recordings found in {path}")
    return sessions


def _cmd_step_count(args: argparse.Namespace) -> None:
    if args.csv.is_dir():
        sessions = _sessions_in(args.csv)

        for session_id, session_paths in sorted(sessions.items()):
            plot_path = Path(args.plot_dir) / f"rec_{session_id}.png" if args.plot_dir else None
            label = f"rec_{session_id}"
            if len(session_paths) > 1:
                label += f" ({len(session_paths)} segments)"
            _run_step_count(session_paths, label, plot_path)
    else:
        plot_path = Path(args.plot) if args.plot else None
        _run_step_count([args.csv], str(args.csv), plot_path)


def _run_step_count(csv_paths: list[Path], label: str, plot_path: Path | None) -> None:
    rec = load_session(csv_paths)
    result = count_steps(rec)

    print(f"file: {label}")
    print(f"  duration: {rec.t[-1]:.1f} s, fs: {rec.fs:.1f} Hz, samples: {rec.n_samples}")
    print(f"  steps detected: {result.n_steps}")
    print(f"  cadence: {result.cadence_spm:.1f} steps/min")
    if result.gt_n_steps is not None:
        print(f"  ground truth steps (dist_mm): {result.gt_n_steps}")

    if plot_path:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_step_count(rec, result, plot_path)
        print(f"  plot saved to {Path(plot_path).with_suffix('.png')} (+ .pdf)")


def _plot_step_count(rec, result, out_path: str, accuracy: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = 5 if rec.has_dist else 4
    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(12, 2.6 * n_rows))

    for i, label in enumerate(["ax", "ay", "az"]):
        axes[i].plot(rec.t, rec.acc[:, i], linewidth=0.6)
        axes[i].plot(
            result.step_times,
            rec.acc[result.step_indices, i],
            "rx",
            label="detected steps" if i == 0 else None,
        )
        axes[i].set_ylabel(f"{label} (g)")
    axes[0].set_title("Raw acceleration, per axis")
    axes[0].legend(loc="upper right")

    axes[3].plot(rec.t, result.filtered_signal, linewidth=0.7)
    axes[3].plot(
        result.step_times,
        result.filtered_signal[result.step_indices],
        "rx",
        label="detected steps",
    )
    axes[3].set_ylabel("|acc|-1g (g)")
    title = f"Step detection, all axes combined ({result.n_steps} steps)"
    if accuracy is not None:
        title += f" — accuracy {accuracy:.2f}"
    axes[3].set_title(title)
    axes[3].legend(loc="upper right")

    if rec.has_dist:
        axes[4].plot(rec.t, rec.dist_mm, linewidth=0.7)
        axes[4].plot(
            result.gt_step_times,
            rec.dist_mm[result.gt_step_indices],
            "gx",
            label="ground truth steps",
        )
        axes[4].axhline(DIST_REST_HIGH_MM, color="gray", linestyle="--", linewidth=0.6)
        axes[4].axhline(DIST_STEP_THRESHOLD_MM, color="gray", linestyle="--", linewidth=0.6)
        axes[4].set_ylabel("cane-tip to ground (mm)")
        axes[4].set_title(f"Ground truth from dist_mm ({result.gt_n_steps} steps)")
        axes[4].legend(loc="upper right")
        axes[4].invert_yaxis()

    axes[-1].set_xlabel("time (s)")

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _cmd_step_accuracy(args: argparse.Namespace) -> None:
    csv_paths = sorted(args.data.glob(args.glob))
    if not csv_paths:
        raise SystemExit(f"no files matching {args.glob!r} found in {args.data}")

    results = compute_window_accuracies(
        csv_paths,
        min_gt_steps=args.min_gt_steps,
        min_gt_cadence_spm=args.min_gt_cadence,
    )
    if not results:
        raise SystemExit(
            f"no window had >= {args.min_gt_steps} ground-truth steps at "
            f">= {args.min_gt_cadence:g} steps/min to score"
        )

    out_path = Path(args.out)
    write_csv(results, out_path)

    import numpy as np

    arr = np.array([r.accuracy for r in results])
    print(
        f"windows scored: {len(results)} of {len(csv_paths)} "
        f"(dropped gt < {args.min_gt_steps} steps or < {args.min_gt_cadence:g} steps/min)"
    )
    print(f"  mean accuracy: {arr.mean():.4f}")
    print(f"  std accuracy:  {arr.std():.4f}")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        scored_files = {r.file: r.accuracy for r in results}
        for path in csv_paths:
            if path.name not in scored_files:
                continue
            rec = load_csv(path)
            result = count_steps(rec)
            plot_path = plot_dir / f"{path.stem}.png"
            _plot_step_count(rec, result, plot_path, accuracy=scored_files[path.name])
        print(f"  {len(scored_files)} plot(s) saved to {plot_dir} (png + pdf)")
    print(f"  csv written to {out_path}")


def _cmd_ae_anomaly(args: argparse.Namespace) -> None:
    config = StepAEConfig(
        window_s=args.window_s,
        target_fs=args.target_fs,
        max_lag_s=args.max_lag_s,
        align_iters=args.align_iters,
        with_dist=args.with_dist,
        anchor=args.anchor,
        hop_s=args.hop_s,
        align_mode=args.align_mode,
        model=args.model,
        hidden=tuple(int(v) for v in args.hidden.split(",")),
        conv_channels=tuple(int(v) for v in args.conv_channels.split(",")),
        kernel_size=args.kernel_size,
        latent=args.latent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        patience=args.patience,
        seed=args.seed,
        threshold_pct=args.threshold_pct,
        score_agg=args.score_agg,
    )

    descriptions = load_descriptions(Path(args.descriptions))
    result = run_ae_anomaly(Path(args.train), Path(args.test), config, descriptions)

    m, o = result.window_metrics, result.oracle_metrics
    n_train = int((result.train_split == "train").sum())
    n_val = int((result.train_split == "val").sum())

    print(f"train: {result.train_windows.n_windows} windows from "
          f"{len(set(result.train_windows.session_ids))} session(s) "
          f"({n_train} train / {n_val} validation)")
    print(f"  model: {config.model}, latent {config.latent}, "
          f"input {result.train_windows.x.shape[1]}x{result.train_windows.x.shape[2]}, "
          f"anchor {config.anchor}, align {config.align_mode}, "
          f"score {config.score_agg}")
    print(f"  alignment branches — train: {result.train_windows.source_counts()}, "
          f"test: {result.test_windows.source_counts()}")
    print(f"  best epoch {result.best_epoch + 1}/{result.train_loss.size}, "
          f"train loss {result.train_loss[result.best_epoch]:.5f}, "
          f"val loss {result.val_loss[result.best_epoch]:.5f}")
    for note in result.notes:
        print(f"  note: {note}")

    print(f"test: {result.test_windows.n_windows} windows from "
          f"{len(result.sessions)} session(s), "
          f"{int(result.scoreable.sum())} labelled "
          f"({int(result.y_true.sum())} anomalous, {result.prevalence:.1%} prevalence)")
    print(f"  threshold (p{config.threshold_pct:g} of held-out normal train error): "
          f"{result.threshold:.5f}")
    print("window-level:")
    print(f"  AUPRC      {result.auprc:.3f}   (chance = prevalence {result.prevalence:.3f})")
    print(f"  ROC-AUC    {result.roc_auc:.3f}   (chance 0.5)")
    print(f"  F1         {m.f1:.3f}")
    print(f"  accuracy   {m.accuracy:.3f}")
    print(f"  precision  {m.precision:.3f}   recall {m.recall:.3f}   "
          f"specificity {m.specificity:.3f}")
    print(f"  confusion  TP {m.tp}  FP {m.fp}  TN {m.tn}  FN {m.fn}")
    print(f"  oracle F1  {o.f1:.3f} (best possible threshold; peeks at test labels)")
    print(f"session-level: {sum(s.correct for s in result.sessions)}/{len(result.sessions)} "
          f"correct ({result.session_accuracy:.0%})")
    for s in sorted(result.sessions, key=lambda s: (s.label, s.session_id)):
        mark = "ok  " if s.correct else "MISS"
        kind = "anomalous" if s.label == 1 else "normal   "
        print(f"  {mark} rec_{s.session_id} {kind} "
              f"flagged {s.n_flagged:3d}/{s.n_labeled:3d} ({s.flag_rate:.2f})"
              + (f"  {s.description}" if s.description else ""))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(result, out_dir / "overview")
    plot_error_histogram(result, out_dir / "error_by_type")
    plot_sessions(result, out_dir / "sessions")
    plot_traces(result, out_dir / "traces")
    plot_examples(result, out_dir / "examples")
    write_session_csv(result, out_dir / "sessions.csv")
    write_window_csv(result, out_dir / "windows.csv")
    write_metrics_csv(result, out_dir / "metrics.csv")
    write_unlabelled_runs_csv(result, out_dir / "unlabelled_runs.csv")
    print(f"plots (png + pdf) and CSVs written to {out_dir}/")


def _cmd_step_ae(args: argparse.Namespace) -> None:
    sessions = _sessions_in(args.csv)

    config = StepAEConfig(
        window_s=args.window_s,
        step_hop=args.step_hop,
        target_fs=args.target_fs,
        max_lag_s=args.max_lag_s,
        align_iters=args.align_iters,
        align_mode=args.align_mode,
        anchor=args.anchor,
        hop_s=args.hop_s,
        model=args.model,
        conv_channels=tuple(int(v) for v in args.conv_channels.split(",")),
        kernel_size=args.kernel_size,
        with_dist=args.with_dist,
        normal_by=args.normal_by,
        normal_ann=tuple(int(v) for v in args.normal_ann.split(",")),
        hidden=tuple(int(v) for v in args.hidden.split(",")),
        latent=args.latent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        patience=args.patience,
        seed=args.seed,
        threshold_pct=args.threshold_pct,
    )

    recs = [(session_id, load_session(paths)) for session_id, paths in sorted(sessions.items())]
    if config.with_dist:
        recs = [(sid, rec) for sid, rec in recs if rec.has_dist]
        if not recs:
            raise SystemExit("--with-dist given but no recording has a dist_mm column")

    result = run_step_ae(
        recs,
        config,
        load_path=Path(args.load) if args.load else None,
        save_path=None if args.load else Path(args.model_out),
    )
    windows = result.windows

    print(f"recordings: {len(recs)} session(s), {sum(r.n_samples for _, r in recs)} samples")
    print(
        f"  windows: {windows.n_windows} of {windows.window_s:g} s "
        f"({windows.window_len} samples at {windows.fs:g} Hz), "
        f"{len(windows.channels)} channels: {', '.join(windows.channels)}"
    )
    print(
        f"  alignment lag: mean |lag| {abs(windows.lags).mean():.1f} samples "
        f"({abs(windows.lags).mean() / windows.fs * 1000:.0f} ms), "
        f"max {abs(windows.lags).max()} samples"
    )
    print(f"  alignment sharpness (template energy): {_sharpness_gain(windows):.2f}x")
    print(f"  alignment mode {result.config.align_mode}, branches {windows.source_counts()}")
    for label in ("train", "val", "candidate"):
        print(f"  {label} windows: {int((result.split == label).sum())}")
    if result.trained:
        print(
            f"  training: best epoch {result.best_epoch + 1}/{len(result.train_loss)}, "
            f"train loss {result.train_loss[result.best_epoch]:.4f}, "
            f"val loss {result.val_loss[result.best_epoch]:.4f}"
        )
    else:
        print(f"  model loaded from {args.load} (not retrained)")
    print(
        f"  threshold (p{result.config.threshold_pct:g} of held-out normal error): "
        f"{result.threshold:.4f}"
    )
    print(f"  flagged: {result.n_flagged_candidates}/{result.n_candidates} candidate window(s), "
          f"{result.n_flagged_normal} normal window(s)")
    worst = int(result.scores.argmax())
    print(
        f"  worst window: rec_{windows.session_ids[worst]} at "
        f"{windows.center_times[worst]:.1f} s, error {result.scores[worst]:.4f}"
    )
    for note in result.notes:
        print(f"  note: {note}")
    if not args.load:
        print(f"  model saved to {args.model_out}")

    if args.plot_dir:
        plot_path = Path(args.plot_dir) / "step_ae.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_step_ae(result, plot_path)
        print(f"  plot saved to {Path(plot_path).with_suffix('.png')} (+ .pdf)")


def _sharpness_gain(windows) -> float:
    """How much alignment concentrated the common pattern.

    Ratio of the mean-signal energy after alignment to before: averaging
    misaligned copies of the same pattern cancels it out, so a ratio above 1
    means the windows now agree on where the pattern is.
    """
    import numpy as np

    before = np.mean(np.mean(windows.align_before, axis=0) ** 2)
    after = np.mean(np.mean(windows.align_after, axis=0) ** 2)
    return float(after / before) if before > 0 else float("nan")


def _plot_step_ae(result, out_path: Path) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = result.windows
    fig, axes = plt.subplots(5, 1, figsize=(12, 2.6 * 5))

    # 1. Did alignment actually sharpen the common pattern?
    t_win = np.arange(windows.window_len) / windows.fs
    for signal, label, color in (
        (windows.align_before, "before alignment", "gray"),
        (windows.align_after, "after alignment", "tab:blue"),
    ):
        mean = signal.mean(axis=0)
        std = signal.std(axis=0)
        axes[0].plot(t_win, mean, linewidth=0.7, color=color, label=label)
        axes[0].fill_between(t_win, mean - std, mean + std, color=color, alpha=0.15)
    axes[0].set_title(
        f"Window alignment: mean +/- std of |acc|-1g "
        f"({_sharpness_gain(windows):.2f}x sharper)"
    )
    axes[0].set_xlabel("time within window (s)")
    axes[0].set_ylabel("|acc|-1g (g)")
    axes[0].legend(loc="upper right")

    # 2. Where the lags landed. Piling up at the bounds means max_lag is too tight.
    max_lag = int(round(result.config.max_lag_s * windows.fs))
    axes[1].hist(windows.lags, bins=range(-max_lag, max_lag + 2), color="tab:blue")
    axes[1].axvline(-max_lag, color="gray", linestyle="--", linewidth=0.6)
    axes[1].axvline(max_lag, color="gray", linestyle="--", linewidth=0.6)
    axes[1].set_title("Estimated alignment lag per window")
    axes[1].set_xlabel("lag (samples)")
    axes[1].set_ylabel("windows")

    # 3. Loss curves - the train/val gap is the overfitting check.
    epochs = np.arange(1, result.train_loss.size + 1)
    axes[2].plot(epochs, result.train_loss, linewidth=0.7, label="train")
    axes[2].plot(epochs, result.val_loss, linewidth=0.7, label="validation")
    axes[2].axvline(result.best_epoch + 1, color="gray", linestyle="--", linewidth=0.6)
    axes[2].set_yscale("log")
    axes[2].set_title("Autoencoder reconstruction loss (MSE)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("loss")
    axes[2].legend(loc="upper right")

    # 4. Score distribution against the threshold.
    normal_scores = result.scores[result.split != "candidate"]
    candidate_scores = result.scores[result.split == "candidate"]
    bins = np.histogram_bin_edges(result.scores, bins=40)
    axes[3].hist(normal_scores, bins=bins, color="tab:blue", label="normal")
    if candidate_scores.size:
        axes[3].hist(candidate_scores, bins=bins, color="tab:red", alpha=0.7, label="candidate")
    axes[3].axvline(result.threshold, color="gray", linestyle="--", linewidth=0.6)
    axes[3].set_title(
        f"Reconstruction error per window (threshold {result.threshold:.4f}, "
        f"{result.n_flagged_candidates}/{result.n_candidates} candidates flagged)"
    )
    axes[3].set_xlabel("mean squared error")
    axes[3].set_ylabel("windows")
    axes[3].legend(loc="upper right")

    # 5. The window the model handled worst, input vs reconstruction.
    worst = int(result.scores.argmax())
    for c in range(len(windows.channels)):
        axes[4].plot(t_win, windows.x[worst, c], linewidth=0.7)
        axes[4].plot(t_win, result.recon[worst, c], "--", linewidth=0.7, color="tab:red")
    axes[4].plot([], [], linewidth=0.7, color="tab:blue", label="input (all channels)")
    axes[4].plot([], [], "--", linewidth=0.7, color="tab:red", label="reconstruction")
    axes[4].set_title(
        f"Worst-reconstructed window: rec_{windows.session_ids[worst]} at "
        f"{windows.center_times[worst]:.1f} s (error {result.scores[worst]:.4f})"
    )
    axes[4].set_xlabel("time within window (s)")
    axes[4].set_ylabel("normalized value")
    axes[4].legend(loc="upper right")

    fig.tight_layout()
    _save_figure(fig, out_path)


# Every ae-anomaly flag whose default is meant to be the StepAEConfig field of
# the same name. Checked rather than trusted: a hardcoded argparse default that
# silently overrides the dataclass is invisible in a diff and shows up only as
# results that do not match the configuration they claim to use.
_SHARED_DEFAULTS = (
    "window_s", "target_fs", "max_lag_s", "align_iters", "anchor", "hop_s",
    "align_mode", "model", "kernel_size", "latent", "epochs", "batch_size",
    "lr", "weight_decay", "val_frac", "patience", "seed", "threshold_pct",
    "score_agg", "with_dist",
)


def _assert_cli_defaults_match_config(parser: argparse.ArgumentParser) -> None:
    drift = [
        f"--{name.replace('_', '-')}: CLI {parser.get_default(name)!r} != config {getattr(_D, name)!r}"
        for name in _SHARED_DEFAULTS
        if parser.get_default(name) != getattr(_D, name)
    ]
    if drift:
        raise SystemExit("CLI defaults have drifted from StepAEConfig:\n  " + "\n  ".join(drift))


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartCane IMU analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    step_parser = sub.add_parser("step-count", help="Count steps from a recording, or all recordings in a directory")
    step_parser.add_argument("csv", type=Path, help="Path to a rec_*.csv IMU log, or a directory of them")
    step_parser.add_argument("--plot", type=str, default=None, help="Save a diagnostic plot to this path (single-file input only)")
    step_parser.add_argument("--plot-dir", type=str, default=None, help="Save a diagnostic plot per input file into this directory")
    step_parser.set_defaults(func=_cmd_step_count)

    acc_parser = sub.add_parser("step-accuracy", help="Score per-window step-count accuracy against dist_mm ground truth")
    acc_parser.add_argument("data", type=Path, help="Directory of rec_*.csv segment files")
    acc_parser.add_argument("--glob", type=str, default="*_ann0.csv", help="Filename pattern to select windows (default *_ann0.csv)")
    acc_parser.add_argument("--min-gt-steps", type=int, default=5, help="Drop windows with fewer ground-truth steps than this (default 5)")
    acc_parser.add_argument("--min-gt-cadence", type=float, default=20.0, help="Drop windows whose ground-truth cadence is below this, in steps/min - they are distance-sensor failures rather than slow walking (default 20)")
    acc_parser.add_argument("--out", type=str, default="out/step_accuracy.csv", help="Output CSV path (default out/step_accuracy.csv)")
    acc_parser.add_argument("--plot-dir", type=str, default=None, help="Save a diagnostic plot per scored window into this directory")
    acc_parser.set_defaults(func=_cmd_step_accuracy)

    an_parser = sub.add_parser("ae-anomaly", help="Train the autoencoder on normal data and evaluate it as an anomaly detector on a labelled test set")
    an_parser.add_argument("--train", type=str, default="data/train", help="Directory of normal training recordings (default data/train)")
    an_parser.add_argument("--test", type=str, default="data/test", help="Directory of labelled test recordings (default data/test)")
    an_parser.add_argument("--descriptions", type=str, default="description.csv", help="Semicolon-separated anomaly descriptions, for the report (default description.csv)")
    an_parser.add_argument("--out-dir", type=str, default="out/ae_anomaly", help="Where to write the report, plots and CSVs (default out/ae_anomaly)")
    an_parser.add_argument("--anchor", choices=["step", "slide"], default=_D.anchor, help="Windowing scheme: slide avoids depending on a step detector that fails on abnormal gait (default slide)")
    an_parser.add_argument("--hop-s", type=float, default=DEFAULT_HOP_S, help=f"Sliding-window spacing in seconds (default {DEFAULT_HOP_S:g})")
    an_parser.add_argument("--align-mode", choices=["none", "xcorr", "step", "mixed"], default=_D.align_mode, help="Phase policy: none leaves windows at their anchor; xcorr correlates against the normal template; step centers the nearest detected step; mixed uses step where one was detected and xcorr only as a fallback (default mixed)")
    an_parser.add_argument("--model", choices=["mlp", "conv"], default=_D.model, help="Autoencoder architecture (default conv)")
    an_parser.add_argument("--score-agg", choices=["mean", "chan_norm", "peak"], default=_D.score_agg, help=f"How per-sample error becomes one score per window (default {_D.score_agg}; chan_norm puts every channel in units of how unusual it is for that channel, which matters once dist_mm is in the mix)")
    an_parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S, help=f"Window length in seconds (default {DEFAULT_WINDOW_S:g})")
    an_parser.add_argument("--target-fs", type=float, default=DEFAULT_TARGET_FS, help=f"Resampling rate in Hz (default {DEFAULT_TARGET_FS:g})")
    an_parser.add_argument("--max-lag-s", type=float, default=DEFAULT_MAX_LAG_S, help=f"Maximum alignment shift in seconds (default {DEFAULT_MAX_LAG_S:g})")
    an_parser.add_argument("--align-iters", type=int, default=3, help="Template refinement passes (default 3)")
    dist_group = an_parser.add_mutually_exclusive_group()
    dist_group.add_argument("--with-dist", dest="with_dist", action="store_true", default=True, help="Include dist_mm (cane-tip height) as an extra channel. On by default; automatically disabled for recordings that predate the sensor")
    dist_group.add_argument("--no-dist", dest="with_dist", action="store_false", help="Exclude dist_mm and score from the IMU alone")
    an_parser.add_argument("--hidden", type=str, default="128,32", help="Comma-separated MLP encoder widths (default 128,32)")
    an_parser.add_argument("--conv-channels", type=str, default="16,32,32", help="Comma-separated conv encoder widths (default 16,32,32)")
    an_parser.add_argument("--kernel-size", type=int, default=7, help="Conv kernel size (default 7)")
    an_parser.add_argument("--latent", type=int, default=_D.latent, help=f"Bottleneck width (default {_D.latent})")
    an_parser.add_argument("--epochs", type=int, default=400, help="Maximum training epochs (default 400)")
    an_parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size (default 32)")
    an_parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default 1e-3)")
    an_parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay (default 1e-5)")
    an_parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of each training session held out for validation and calibration (default 0.2)")
    an_parser.add_argument("--patience", type=int, default=60, help="Early-stopping patience in epochs (default 60)")
    an_parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    an_parser.add_argument("--threshold-pct", type=float, default=_D.threshold_pct, help="Percentile of held-out normal error used as the threshold, i.e. the false-alarm budget: p95 accepts a 5%% false-alarm rate on normal gait (default 95)")
    an_parser.set_defaults(func=_cmd_ae_anomaly)

    ae_parser = sub.add_parser("step-ae", help="Train an autoencoder over aligned step windows and score reconstruction error")
    ae_parser.add_argument("csv", type=Path, help="Path to a rec_*.csv IMU log, or a directory of them")
    ae_parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S, help=f"Window length in seconds (default {DEFAULT_WINDOW_S:g}, about three steps)")
    ae_parser.add_argument("--step-hop", type=int, default=1, help="Anchor a window on every Nth detected step (default 1)")
    ae_parser.add_argument("--target-fs", type=float, default=DEFAULT_TARGET_FS, help=f"Resampling rate in Hz (default {DEFAULT_TARGET_FS:g})")
    ae_parser.add_argument("--max-lag-s", type=float, default=DEFAULT_MAX_LAG_S, help=f"Maximum alignment shift in seconds (default {DEFAULT_MAX_LAG_S:g})")
    ae_parser.add_argument("--anchor", choices=["step", "slide"], default=_D.anchor, help=f"Windowing scheme (default {_D.anchor}; step ties how much evidence a recording yields to a step detector that fails on abnormal gait)")
    ae_parser.add_argument("--hop-s", type=float, default=DEFAULT_HOP_S, help=f"Sliding-window spacing in seconds (default {DEFAULT_HOP_S:g})")
    ae_parser.add_argument("--align-mode", choices=["none", "xcorr", "step", "mixed"], default=_D.align_mode, help=f"Phase policy (default {_D.align_mode}; step placement where a step was detected, correlation only as a fallback)")
    ae_parser.add_argument("--model", choices=["mlp", "conv"], default=_D.model, help=f"Autoencoder architecture (default {_D.model})")
    ae_parser.add_argument("--conv-channels", type=str, default="16,32,32", help="Comma-separated conv encoder widths (default 16,32,32)")
    ae_parser.add_argument("--kernel-size", type=int, default=7, help="Conv kernel size (default 7)")
    ae_parser.add_argument("--align-iters", type=int, default=3, help="Template refinement passes, used only when fitting an xcorr template from scratch (default 3)")
    ae_dist_group = ae_parser.add_mutually_exclusive_group()
    ae_dist_group.add_argument("--with-dist", dest="with_dist", action="store_true", default=True, help="Include dist_mm (cane-tip height) as an extra channel. On by default; automatically disabled for recordings that predate the sensor")
    ae_dist_group.add_argument("--no-dist", dest="with_dist", action="store_false", help="Exclude dist_mm and score from the IMU alone")
    ae_parser.add_argument("--normal-by", choices=["event", "ann"], default="event", help="Label source marking a window as a candidate anomaly (default event)")
    ae_parser.add_argument("--normal-ann", type=str, default="0", help="Comma-separated ann values that are normal end to end, for --normal-by ann (default 0). ann-1 is always resolved per window against the event marker, since that is what the annotation means")
    ae_parser.add_argument("--hidden", type=str, default="128,32", help="Comma-separated encoder widths (default 128,32)")
    ae_parser.add_argument("--latent", type=int, default=_D.latent, help=f"Bottleneck width (default {_D.latent})")
    ae_parser.add_argument("--epochs", type=int, default=400, help="Maximum training epochs (default 400)")
    ae_parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size (default 32)")
    ae_parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default 1e-3)")
    ae_parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay (default 1e-5)")
    ae_parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of normal windows held out for validation (default 0.2)")
    ae_parser.add_argument("--patience", type=int, default=60, help="Early-stopping patience in epochs (default 60)")
    ae_parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    ae_parser.add_argument("--threshold-pct", type=float, default=_D.threshold_pct, help=f"Percentile of held-out normal error used as the anomaly threshold, i.e. the false-alarm budget (default {_D.threshold_pct:g})")
    ae_parser.add_argument("--model-out", type=str, default="out/step_ae/model.pt", help="Where to save the trained model (default out/step_ae/model.pt)")
    ae_parser.add_argument("--load", type=str, default=None, help="Score with a saved model instead of training")
    ae_parser.add_argument("--plot-dir", type=str, default=None, help="Save a diagnostic plot into this directory")
    ae_parser.set_defaults(func=_cmd_step_ae)

    _assert_cli_defaults_match_config(an_parser)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
