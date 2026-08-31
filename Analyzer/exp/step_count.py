"""Experiment 1: count steps from the cane's acceleration signal.

Approach: band-pass filter the acceleration-magnitude signal around the
typical walking cadence range, then find peaks in the filtered envelope,
each peak corresponding to one cane strike / step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from utils.io import ImuRecording

# Walking cadence is roughly 0.5-3 Hz (30-180 steps/min). Cane taps happen
# at a similar rate, so we band-pass the acceleration magnitude there to
# suppress gravity/drift and high-frequency noise before peak-picking.
_BAND_LOW_HZ = 0.5
_BAND_HIGH_HZ = 3.5
_MIN_STEP_INTERVAL_S = 0.25  # cap cadence at 240 steps/min
_MAD_TO_STD = 1.4826  # scales MAD to a std-equivalent for a normal noise floor
_HEIGHT_MAD_MULTIPLIER = 5.0
# Floor for the detection threshold, in g. A near-motionless recording has a
# near-zero MAD, which would otherwise let filtfilt's edge transients (or
# residual sensor noise) get picked up as "steps".
_MIN_HEIGHT_G = 0.02


@dataclass
class StepCountResult:
    n_steps: int
    step_indices: np.ndarray  # sample index of each detected step
    step_times: np.ndarray  # seconds, time of each detected step
    step_intervals: np.ndarray  # seconds between consecutive steps
    cadence_spm: float  # mean steps per minute
    filtered_signal: np.ndarray  # band-passed |acc| used for detection


def _bandpass(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(N=4, Wn=[low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def count_steps(rec: ImuRecording) -> StepCountResult:
    mag = rec.acc_mag - np.mean(rec.acc_mag)
    filtered = _bandpass(mag, rec.fs, _BAND_LOW_HZ, _BAND_HIGH_HZ)

    min_distance = max(1, int(_MIN_STEP_INTERVAL_S * rec.fs))
    # A global std threshold shrinks as an idle tail dilutes the recording,
    # which would let noise start false-triggering in long logs. MAD instead
    # tracks the quiet-period noise floor (robust as long as walking is a
    # minority of the recording), so the threshold stays meaningful either way.
    mad = np.median(np.abs(filtered - np.median(filtered)))
    height = max(_HEIGHT_MAD_MULTIPLIER * _MAD_TO_STD * mad, _MIN_HEIGHT_G)

    peaks, _ = find_peaks(filtered, height=height, distance=min_distance)
    step_times = rec.t[peaks]
    step_intervals = np.diff(step_times)

    cadence_spm = 0.0
    if step_times.size >= 2:
        duration_min = (step_times[-1] - step_times[0]) / 60.0
        if duration_min > 0:
            cadence_spm = (step_times.size - 1) / duration_min

    return StepCountResult(
        n_steps=step_times.size,
        step_indices=peaks,
        step_times=step_times,
        step_intervals=step_intervals,
        cadence_spm=cadence_spm,
        filtered_signal=filtered,
    )
