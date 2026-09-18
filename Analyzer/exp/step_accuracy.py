"""Per-window step-count accuracy against the dist_mm ground truth.

Accuracy per window is 1 - |detected - ground_truth| / ground_truth, clipped
at 0. Windows are dropped if they have too few ground-truth steps
(`min_gt_steps`) or an implausibly low ground-truth cadence
(`min_gt_cadence_spm`) - the latter catches a distance-sensor failure (tip
reading stuck away from the resting band) rather than a slow walk.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exp.step_count import count_steps
from utils.io import load_csv


@dataclass
class WindowAccuracy:
    file: str
    gt_steps: int
    gt_cadence_spm: float
    detected_steps: int
    abs_error: int
    accuracy: float


def compute_window_accuracies(
    csv_paths: list[Path], min_gt_steps: int = 5, min_gt_cadence_spm: float = 20.0
) -> list[WindowAccuracy]:
    results = []
    for path in sorted(csv_paths):
        rec = load_csv(path)
        if not rec.has_dist:
            continue
        result = count_steps(rec)
        gt = result.gt_n_steps or 0
        if gt < min_gt_steps:
            continue
        duration_s = float(rec.t[-1])
        gt_cadence = gt / duration_s * 60.0 if duration_s > 0 else 0.0
        if gt_cadence < min_gt_cadence_spm:
            continue
        detected = result.n_steps
        abs_error = abs(detected - gt)
        accuracy = max(0.0, 1.0 - abs_error / gt)
        results.append(
            WindowAccuracy(
                file=path.name,
                gt_steps=gt,
                gt_cadence_spm=gt_cadence,
                detected_steps=detected,
                abs_error=abs_error,
                accuracy=accuracy,
            )
        )
    return results


def write_csv(results: list[WindowAccuracy], out_path: Path) -> None:
    accuracies = np.array([r.accuracy for r in results])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["file", "gt_steps", "gt_cadence_spm", "detected_steps", "abs_error", "accuracy"]
        )
        for r in results:
            writer.writerow(
                [
                    r.file,
                    r.gt_steps,
                    f"{r.gt_cadence_spm:.1f}",
                    r.detected_steps,
                    r.abs_error,
                    f"{r.accuracy:.4f}",
                ]
            )
        writer.writerow([])
        writer.writerow(["summary", "n_windows", "mean_accuracy", "std_accuracy"])
        writer.writerow(
            [
                "",
                len(results),
                f"{accuracies.mean():.4f}" if accuracies.size else "",
                f"{accuracies.std():.4f}" if accuracies.size else "",
            ]
        )
