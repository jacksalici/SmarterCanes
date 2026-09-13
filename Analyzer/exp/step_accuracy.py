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

Windows whose ground-truth *rate* is implausibly low are dropped too
(`min_gt_cadence_spm`, default 20). A count can clear `min_gt_steps` and still
be a ground-truth failure rather than a slow walk: the distance sensor detects
a step as a rising crossing back out of the resting band, so whenever the tip
reading never returns to that band - the sensor lost the ground, the cane was
carried, the baseline drifted - real steps become invisible and the count
collapses. Those windows are recognisable because the tip reads as lifted for
far longer than a swing (2-32 s against 1.0-1.5 s on sound windows), and
scoring the detector against them measures the sensor, not the detector.
Scoreable windows sit at 22-34 steps/min, so a floor of 20 steps/min - one
step per three seconds, slower than any continuous walk here - separates the
two without cutting into real gait.
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
