"""Per-window step-count accuracy against the dist_mm ground truth.

Each `ann-0` file in `data/` is one independent 30-second segment. For every
such segment, `count_steps` gives both the detected step count (from
acceleration) and the ground-truth count (from `dist_mm`); this compares the
two per window rather than across a stitched session, since the annotation
and segmentation are per-file here.

Accuracy per window is 1 - |detected - ground_truth| / ground_truth, i.e. how
close the detected count came to ground truth relative to its size (clipped
at 0 so a wildly-off window doesn't go negative). Windows with too few
ground-truth steps (`min_gt_steps`, default 5) are dropped entirely: a low
count makes both the relative error noisy and the window itself too short a
stretch of gait to say much about detector accuracy.
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
    detected_steps: int
    abs_error: int
    accuracy: float


def compute_window_accuracies(
    csv_paths: list[Path], min_gt_steps: int = 5
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
        detected = result.n_steps
        abs_error = abs(detected - gt)
        accuracy = max(0.0, 1.0 - abs_error / gt)
        results.append(
            WindowAccuracy(
                file=path.name,
                gt_steps=gt,
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
        writer.writerow(["file", "gt_steps", "detected_steps", "abs_error", "accuracy"])
        for r in results:
            writer.writerow(
                [r.file, r.gt_steps, r.detected_steps, r.abs_error, f"{r.accuracy:.4f}"]
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
