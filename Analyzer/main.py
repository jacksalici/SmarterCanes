#!/usr/bin/env python3
"""SmartCane IMU analyzer entry point.

Usage:
    python main.py step-count data/rec_00004.csv [--plot out.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exp.step_count import count_steps
from utils.io import load_csv


def _cmd_step_count(args: argparse.Namespace) -> None:
    rec = load_csv(args.csv)
    result = count_steps(rec)

    print(f"file: {rec.source}")
    print(f"duration: {rec.t[-1]:.1f} s, fs: {rec.fs:.1f} Hz, samples: {rec.n_samples}")
    print(f"steps detected: {result.n_steps}")
    print(f"cadence: {result.cadence_spm:.1f} steps/min")

    if args.plot:
        _plot_step_count(rec, result, args.plot)
        print(f"plot saved to {args.plot}")


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

    step_parser = sub.add_parser("step-count", help="Count steps from a recording")
    step_parser.add_argument("csv", type=Path, help="Path to a rec_*.csv IMU log")
    step_parser.add_argument("--plot", type=str, default=None, help="Save a diagnostic plot to this path")
    step_parser.set_defaults(func=_cmd_step_count)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
