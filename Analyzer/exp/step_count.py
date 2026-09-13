"""Experiment 1: count steps from the cane's acceleration signal.

Approach: a cane tap/strike is a sharp, broadband mechanical shock, not a
smooth rhythmic oscillation, so band-passing it around a cadence band (the
previous approach) spreads one impulse's energy into a ringing filter
response with several peaks, over-counting steps by roughly 3x. Instead,
this looks for large isolated excursions of the raw acceleration magnitude
away from 1 g, using all three axes: a real strike is a big enough shock to
show up above the walking/handling noise floor on the combined magnitude,
however it's oriented. The height threshold is set from the noise floor of
this specific recording (mean + k*std of |acc|-1g) rather than a fixed
value, and calibrated against ground-truth (dist_mm) step counts.
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

# Refractory period between strikes. Ground-truth step intervals on these
# recordings have a median of 1.97 s and a 1st percentile of 0.98 s, so cane
# cadence tops out near 60 steps/min - nothing like the 150 steps/min the
# previous 0.4 s gate allowed. That slack was the detector's main error
# source: one strike rings for a few hundred milliseconds, and every rebound
# past the height threshold was counted as another step. Widening the gate
# costs the fastest ~2% of real steps and removes far more double-counts than
# it merges.
_STEP_MIN_INTERVAL_S = 1.15  # cap cadence at ~52 steps/min
# Detection height = mean + k * std of |acc|-1g over the whole recording.
# Calibrated against ground-truth (dist_mm) step counts on real recordings.
_STEP_HEIGHT_SIGMA_K = 3.2
# A strike also has to stand clear of its own surroundings by this fraction of
# the detection height. Height alone accepts a rebound that never falls back to
# the noise floor; prominence is what distinguishes one shock from its ringing.
_STEP_PROMINENCE_FRAC = 0.5
# Absolute floor, in g. A near-motionless recording has near-zero noise, so
# std alone could let residual sensor noise get picked up as "steps". Real
# strikes in this dataset peak at 0.59-1.31 g of deviation, while a motionless
# recording peaks around 0.03 g, so a floor in between rejects the second
# without ever binding on the first (it binds on 0 of the 45 scoreable
# windows). The previous 0.02 g floor was low enough to report "steps" in a
# recording with no motion at all.
_MIN_HEIGHT_G = 0.30

# Band thresholds for the distance sensor live in utils.io, because the
# windowing preprocessor needs the same error floor; re-exported here so
# existing callers keep importing them from this module.
_DIST_MIN_INTERVAL_S = 0.25  # cap cadence at 240 steps/min
# A swing's mid-flight sensor jitter can dip back into the resting band for
# a sample or two before continuing up - a bare touch isn't enough evidence
# the cane actually landed, so require it to stay there for a stretch
# before re-arming (a real return-to-rest holds far longer than this).
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
    """Detect steps from the distance sensor.

    The reading quantizes into three bands rather than behaving like a
    clean waveform: 90-110 mm is resting noise (cane planted), a rise past
    118 mm means the cane tip has actually moved away from the ground (a
    step), and anything below 90 mm is a sensor measurement error. So a
    step is each *rising* crossing of the 118 mm threshold, latched with
    hysteresis - it can't fire again until the reading has come back down
    and *stayed* in the resting band for a stretch (not just touched it,
    which is often mid-swing sensor jitter rather than the cane landing) -
    rather than a peak/trough search on a filtered signal, which is fragile
    against this sensor's jitter.
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
