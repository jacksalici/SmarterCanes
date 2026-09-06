"""Experiment 1: count steps from the cane's acceleration signal.

Approach: band-pass filter the acceleration magnitude (transverse axes
only, see `transverse_acc_mag`) around the typical walking cadence range,
then find peaks in the filtered envelope, each peak corresponding to one
cane strike / step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from utils.io import ImuRecording

# Walking cadence is roughly 0.5-3 Hz (30-180 steps/min). Cane taps happen
# at a similar rate, so we band-pass the acceleration signal there to
# suppress gravity/drift and high-frequency noise before peak-picking.
_BAND_LOW_HZ = 0.5
_BAND_HIGH_HZ = 3.5
_MIN_STEP_INTERVAL_S = 0.25  # cap cadence at 240 steps/min
# Fraction of the median candidate-peak height used as the detection
# threshold. Based on the peaks themselves (not the whole signal), so it
# works both for long recordings with a mostly-idle tail and for short
# recordings that are almost entirely gait activity.
_HEIGHT_MEDIAN_FRACTION = 0.4
# Absolute floor, in g. A near-motionless recording has near-zero peaks,
# which would otherwise let filtfilt's edge transients (or residual sensor
# noise) get picked up as "steps".
_MIN_HEIGHT_G = 0.02

# Ground-truth (dist_mm) step detector: reading is between the cane base and
# the ground, and behaves as three bands rather than a smooth waveform.
# 110-120 mm is the resting/noise band (cane planted, sensor jitter only);
# above 125 mm the cane tip has actually lifted off/away from the ground,
# i.e. a step; below 110 mm is a sensor measurement error, not a real
# reading. These are empirical calibration values for this sensor/mounting.
DIST_ERROR_FLOOR_MM = 110.0
DIST_REST_HIGH_MM = 120.0
DIST_STEP_THRESHOLD_MM = 125.0


@dataclass
class StepCountResult:
    n_steps: int
    step_indices: np.ndarray  # sample index of each detected step
    step_times: np.ndarray  # seconds, time of each detected step
    step_intervals: np.ndarray  # seconds between consecutive steps
    cadence_spm: float  # mean steps per minute
    filtered_signal: np.ndarray  # band-passed |acc| used for detection
    gt_n_steps: int | None = None  # ground-truth step count, from dist_mm
    gt_step_indices: np.ndarray | None = None  # sample index of each ground-truth step
    gt_step_times: np.ndarray | None = None  # seconds, time of each ground-truth step
    gt_filtered_signal: np.ndarray | None = None  # error-floor-clipped dist_mm used for detection


def _detect_gt_steps(rec: ImuRecording) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect steps from the distance sensor.

    The reading quantizes into three bands rather than behaving like a
    clean waveform: 110-120 mm is resting noise (cane planted), a rise past
    125 mm means the cane tip has actually moved away from the ground (a
    step), and anything below 110 mm is a sensor measurement error. So a
    step is each *rising* crossing of the 125 mm threshold, latched with
    hysteresis - it can't fire again until the reading has come back down
    to the resting band - rather than a peak/trough search on a filtered
    signal, which is fragile against this sensor's jitter.
    """
    dist = np.where(rec.dist_mm < DIST_ERROR_FLOOR_MM, DIST_ERROR_FLOOR_MM, rec.dist_mm)
    min_gap = max(1, int(_MIN_STEP_INTERVAL_S * rec.fs))

    peaks = []
    armed = True  # can trigger a step; disarmed until back in the resting band
    last_idx = -min_gap
    for i, d in enumerate(dist):
        if armed and d > DIST_STEP_THRESHOLD_MM:
            if i - last_idx >= min_gap:
                peaks.append(i)
                last_idx = i
            armed = False
        elif not armed and d <= DIST_REST_HIGH_MM:
            armed = True
    peaks = np.array(peaks, dtype=int)

    return peaks, rec.t[peaks], dist


def _bandpass(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(N=4, Wn=[low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def transverse_acc_mag(acc: np.ndarray) -> np.ndarray:
    """Acceleration magnitude using only the axes transverse to gravity.

    The axis most aligned with gravity (mean reading closest to +-1 g) is
    almost pure DC - the cane doesn't rotate enough for tap/swing motion to
    show up there - so folding it into the norm just dilutes the transient
    the detector is looking for. Excluding it consistently gives a much
    stronger gait-band signal than the full 3-axis magnitude.
    """
    gravity_axis = int(np.argmin(np.abs(np.abs(np.mean(acc, axis=0)) - 1.0)))
    other_axes = [i for i in range(acc.shape[1]) if i != gravity_axis]
    return np.linalg.norm(acc[:, other_axes], axis=1)


def count_steps(rec: ImuRecording) -> StepCountResult:
    mag = transverse_acc_mag(rec.acc)
    mag -= np.mean(mag)
    filtered = _bandpass(mag, rec.fs, _BAND_LOW_HZ, _BAND_HIGH_HZ)

    min_distance = max(1, int(_MIN_STEP_INTERVAL_S * rec.fs))

    candidates, _ = find_peaks(filtered, distance=min_distance)
    if candidates.size:
        height = max(_HEIGHT_MEDIAN_FRACTION * np.median(filtered[candidates]), _MIN_HEIGHT_G)
        peaks = candidates[filtered[candidates] >= height]
    else:
        peaks = candidates

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
        filtered_signal=filtered,
        gt_n_steps=gt_n_steps,
        gt_step_indices=gt_step_indices,
        gt_step_times=gt_step_times,
        gt_filtered_signal=gt_filtered,
    )
