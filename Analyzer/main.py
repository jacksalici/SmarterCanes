#!/usr/bin/env python3
"""SmartCane IMU analyzer entry point.

Usage:
    python main.py step-count data/rec_00004.csv [--plot out.png]
    python main.py step-count data [--plot-dir out/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exp.step_count import count_steps
from utils.io import load_csv


def _cmd_step_count(args: argparse.Namespace) -> None:
    if args.csv.is_dir():
        csv_paths = sorted(args.csv.glob("*.csv"))
        if not csv_paths:
            raise SystemExit(f"no .csv files found in {args.csv}")
    else:
        csv_paths = [args.csv]

    for csv_path in csv_paths:
        plot_path = None
        if args.plot_dir:
            plot_path = Path(args.plot_dir) / f"{csv_path.stem}.png"
        elif args.plot and len(csv_paths) == 1:
            plot_path = Path(args.plot)

        _run_step_count(csv_path, plot_path)


def _run_step_count(csv_path: Path, plot_path: Path | None) -> None:
    rec = load_csv(csv_path)
    result = count_steps(rec)

    print(f"file: {rec.source}")
    print(f"  duration: {rec.t[-1]:.1f} s, fs: {rec.fs:.1f} Hz, samples: {rec.n_samples}")
    print(f"  steps detected: {result.n_steps}")
    print(f"  cadence: {result.cadence_spm:.1f} steps/min")

    if plot_path:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_step_count(rec, result, plot_path)
        print(f"  plot saved to {plot_path}")


def _plot_step_count(rec, result, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    axes[0].plot(rec.t, rec.acc_mag, linewidth=0.7)
    axes[0].set_ylabel("|acc| (g)")
    axes[0].set_title("Raw acceleration magnitude")

    axes[1].plot(rec.t, result.filtered_signal, linewidth=0.7)
    axes[1].plot(
        result.step_times,
        result.filtered_signal[result.step_indices],
        "rx",
        label="detected steps",
    )
    axes[1].set_ylabel("filtered |acc| (g)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title(f"Step detection ({result.n_steps} steps)")
    axes[1].legend(loc="upper right")

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
