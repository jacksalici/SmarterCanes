"""Experiment 1: count steps as isolated shocks in |acc| deviation from 1 g.

Detection height is mean + k*std of |acc|-1g for the specific recording,
calibrated against ground-truth (dist_mm) step counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from utils.io import (
    DIST_ERROR_FLOOR_MM,
    DIST_REST_HIGH_MM,
    DIST_STEP_THRESHOLD_MM,
    ImuRecording,
)

__all__ = [
    "DIST_ERROR_FLOOR_MM",
    "DIST_REST_HIGH_MM",
    "DIST_STEP_THRESHOLD_MM",
    "StepCountResult",
    "count_steps",
]

# Refractory period between strikes: caps cadence at ~52 steps/min, wide
# enough to reject a strike's own ringing.
_STEP_MIN_INTERVAL_S = 1.15
# Detection height = mean + k * std of |acc|-1g, calibrated against dist_mm.
_STEP_HEIGHT_SIGMA_K = 3.2
# Fraction of the detection height a strike must stand clear of its
# surroundings by, to distinguish one shock from its own ringing.
_STEP_PROMINENCE_FRAC = 0.5
# Absolute floor in g, so a near-motionless recording (~0.03 g noise) can't
# trigger on sensor noise; real strikes peak at 0.59-1.31 g.
_MIN_HEIGHT_G = 0.30

# Distance-sensor thresholds live in utils.io (shared with the windowing
# preprocessor); re-exported here for existing callers.
_DIST_MIN_INTERVAL_S = 0.25  # cap cadence at 240 steps/min
# Samples the reading must hold in the resting band before re-arming, so
# mid-swing jitter can't pass for a return-to-rest.
_DIST_REARM_HOLD_S = 0.1


@dataclass
class StepCountResult:
    n_steps: int
    step_indices: np.ndarray  # sample index of each detected step
    step_times: np.ndarray  # seconds, time of each detected step
    step_intervals: np.ndarray  # seconds between consecutive steps
    cadence_spm: float  # mean steps per minute
    filtered_signal: np.ndarray  # |acc|-1g deviation used for detection
    gt_n_steps: int | None = None  # ground-truth step count, from dist_mm
    gt_step_indices: np.ndarray | None = None  # sample index of each ground-truth step
    gt_step_times: np.ndarray | None = None  # seconds, time of each ground-truth step
    gt_filtered_signal: np.ndarray | None = None  # error-floor-clipped dist_mm used for detection


def _detect_gt_steps(rec: ImuRecording) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect steps as rising crossings of the tip-lift threshold, latched
    with hysteresis until the reading holds in the resting band again.
    """
    dist = np.where(rec.dist_mm < DIST_ERROR_FLOOR_MM, DIST_ERROR_FLOOR_MM, rec.dist_mm)
    min_gap = max(1, int(_DIST_MIN_INTERVAL_S * rec.fs))
    rearm_hold = max(1, int(round(_DIST_REARM_HOLD_S * rec.fs)))

    peaks = []
    armed = True  # can trigger a step; disarmed until back in the resting band
    rest_streak = 0
    last_idx = -min_gap
    for i, d in enumerate(dist):
        if armed:
            if d > DIST_STEP_THRESHOLD_MM:
                if i - last_idx >= min_gap:
                    peaks.append(i)
                    last_idx = i
                armed = False
                rest_streak = 0
        elif d <= DIST_REST_HIGH_MM:
            rest_streak += 1
            if rest_streak >= rearm_hold:
                armed = True
        else:
            rest_streak = 0
    peaks = np.array(peaks, dtype=int)

    return peaks, rec.t[peaks], dist


def count_steps(rec: ImuRecording) -> StepCountResult:
    dev = np.abs(rec.acc_mag - 1.0)
    min_distance = max(1, int(_STEP_MIN_INTERVAL_S * rec.fs))
    height = max(np.mean(dev) + _STEP_HEIGHT_SIGMA_K * np.std(dev), _MIN_HEIGHT_G)

    peaks, _ = find_peaks(
        dev,
        height=height,
        distance=min_distance,
        prominence=_STEP_PROMINENCE_FRAC * height,
    )

    step_times = rec.t[peaks]
    step_intervals = np.diff(step_times)

    cadence_spm = 0.0
    if step_times.size >= 2:
        duration_min = (step_times[-1] - step_times[0]) / 60.0
        if duration_min > 0:
            cadence_spm = (step_times.size - 1) / duration_min

    gt_step_indices = gt_step_times = gt_filtered = None
    gt_n_steps = None
    if rec.has_dist:
        gt_step_indices, gt_step_times, gt_filtered = _detect_gt_steps(rec)
        gt_n_steps = gt_step_times.size

    return StepCountResult(
        n_steps=step_times.size,
        step_indices=peaks,
        step_times=step_times,
        step_intervals=step_intervals,
        cadence_spm=cadence_spm,
        filtered_signal=dev,
        gt_n_steps=gt_n_steps,
        gt_step_indices=gt_step_indices,
        gt_step_times=gt_step_times,
        gt_filtered_signal=gt_filtered,
    )
