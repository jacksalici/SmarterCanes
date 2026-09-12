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
    DEFAULT_MAX_LAG_S,
    DEFAULT_TARGET_FS,
    DEFAULT_WINDOW_S,
    StepAEConfig,
    run_step_ae,
)
from exp.step_count import (
    DIST_REST_HIGH_MM,
    DIST_STEP_THRESHOLD_MM,
    count_steps,
)
from utils.io import group_sessions, load_session


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
        print(f"  plot saved to {plot_path}")


def _plot_step_count(rec, result, out_path: str) -> None:
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
    axes[3].set_title(f"Step detection, all axes combined ({result.n_steps} steps)")
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
    fig.savefig(out_path, dpi=150)


def _cmd_step_ae(args: argparse.Namespace) -> None:
    sessions = _sessions_in(args.csv)

    config = StepAEConfig(
        window_s=args.window_s,
        step_hop=args.step_hop,
        target_fs=args.target_fs,
        max_lag_s=args.max_lag_s,
        align_iters=args.align_iters,
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
        print(f"  plot saved to {plot_path}")


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
    fig.savefig(out_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartCane IMU analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    step_parser = sub.add_parser("step-count", help="Count steps from a recording, or all recordings in a directory")
    step_parser.add_argument("csv", type=Path, help="Path to a rec_*.csv IMU log, or a directory of them")
    step_parser.add_argument("--plot", type=str, default=None, help="Save a diagnostic plot to this path (single-file input only)")
    step_parser.add_argument("--plot-dir", type=str, default=None, help="Save a diagnostic plot per input file into this directory")
    step_parser.set_defaults(func=_cmd_step_count)

    ae_parser = sub.add_parser("step-ae", help="Train an autoencoder over aligned step windows and score reconstruction error")
    ae_parser.add_argument("csv", type=Path, help="Path to a rec_*.csv IMU log, or a directory of them")
    ae_parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S, help=f"Window length in seconds (default {DEFAULT_WINDOW_S:g}, about three steps)")
    ae_parser.add_argument("--step-hop", type=int, default=1, help="Anchor a window on every Nth detected step (default 1)")
    ae_parser.add_argument("--target-fs", type=float, default=DEFAULT_TARGET_FS, help=f"Resampling rate in Hz (default {DEFAULT_TARGET_FS:g})")
    ae_parser.add_argument("--max-lag-s", type=float, default=DEFAULT_MAX_LAG_S, help=f"Maximum alignment shift in seconds (default {DEFAULT_MAX_LAG_S:g})")
    ae_parser.add_argument("--align-iters", type=int, default=3, help="Template refinement passes (default 3)")
    ae_parser.add_argument("--with-dist", action="store_true", help="Include dist_mm as an extra channel (it is the step-detection ground truth, so off by default)")
    ae_parser.add_argument("--normal-by", choices=["event", "ann"], default="event", help="Label source marking a window as a candidate anomaly (default event)")
    ae_parser.add_argument("--normal-ann", type=str, default="-1", help="Comma-separated ann values treated as normal, for --normal-by ann (default -1)")
    ae_parser.add_argument("--hidden", type=str, default="128,32", help="Comma-separated encoder widths (default 128,32)")
    ae_parser.add_argument("--latent", type=int, default=8, help="Bottleneck width (default 8)")
    ae_parser.add_argument("--epochs", type=int, default=400, help="Maximum training epochs (default 400)")
    ae_parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size (default 32)")
    ae_parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default 1e-3)")
    ae_parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay (default 1e-5)")
    ae_parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of normal windows held out for validation (default 0.2)")
    ae_parser.add_argument("--patience", type=int, default=60, help="Early-stopping patience in epochs (default 60)")
    ae_parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    ae_parser.add_argument("--threshold-pct", type=float, default=99.0, help="Percentile of held-out normal error used as the anomaly threshold (default 99)")
    ae_parser.add_argument("--model-out", type=str, default="out/step_ae/model.pt", help="Where to save the trained model (default out/step_ae/model.pt)")
    ae_parser.add_argument("--load", type=str, default=None, help="Score with a saved model instead of training")
    ae_parser.add_argument("--plot-dir", type=str, default=None, help="Save a diagnostic plot into this directory")
    ae_parser.set_defaults(func=_cmd_step_ae)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
