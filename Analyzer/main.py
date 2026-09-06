#!/usr/bin/env python3
"""SmartCane IMU analyzer entry point.

Usage:
    python main.py step-count data/rec_00004.csv [--plot out.png]
    python main.py step-count data [--plot-dir out/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exp.step_count import (
    DIST_REST_HIGH_MM,
    DIST_STEP_THRESHOLD_MM,
    count_steps,
    transverse_acc_mag,
)
from utils.io import group_sessions, load_session


def _cmd_step_count(args: argparse.Namespace) -> None:
    if args.csv.is_dir():
        csv_paths = sorted(args.csv.glob("*.csv"))
        if not csv_paths:
            raise SystemExit(f"no .csv files found in {args.csv}")

        sessions = group_sessions(csv_paths)
        if not sessions:
            raise SystemExit(f"no rec_*.csv recordings found in {args.csv}")

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

    n_rows = 3 if rec.has_dist else 2
    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(12, 3 * n_rows))

    axes[0].plot(rec.t, transverse_acc_mag(rec.acc), linewidth=0.7)
    axes[0].set_ylabel("transverse |acc| (g)")
    axes[0].set_title("Raw acceleration magnitude (gravity axis excluded)")

    axes[1].plot(rec.t, result.filtered_signal, linewidth=0.7)
    axes[1].plot(
        result.step_times,
        result.filtered_signal[result.step_indices],
        "rx",
        label="detected steps",
    )
    axes[1].set_ylabel("filtered |acc| (g)")
    axes[1].set_title(f"Step detection ({result.n_steps} steps)")
    axes[1].legend(loc="upper right")

    if rec.has_dist:
        axes[2].plot(rec.t, rec.dist_mm, linewidth=0.7)
        axes[2].plot(
            result.gt_step_times,
            rec.dist_mm[result.gt_step_indices],
            "gx",
            label="ground truth steps",
        )
        axes[2].axhline(DIST_REST_HIGH_MM, color="gray", linestyle="--", linewidth=0.6)
        axes[2].axhline(DIST_STEP_THRESHOLD_MM, color="gray", linestyle="--", linewidth=0.6)
        axes[2].set_ylabel("cane-tip to ground (mm)")
        axes[2].set_title(f"Ground truth from dist_mm ({result.gt_n_steps} steps)")
        axes[2].legend(loc="upper right")
        axes[2].invert_yaxis()

    axes[-1].set_xlabel("time (s)")

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
