"""Loading of raw IMU CSV recordings produced by the SmartCane firmware."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Raw units used by the firmware log.
_MG_PER_G = 1000.0
_MDPS_PER_DPS = 1000.0

_N_COLUMNS = 7  # t_ms, ax_mg, ay_mg, az_mg, gx_mdps, gy_mdps, gz_mdps


@dataclass
class ImuRecording:
    """A single IMU recording, converted to physical units."""

    t: np.ndarray  # seconds, starting at 0
    acc: np.ndarray  # (N, 3) in g, columns [x, y, z]
    gyro: np.ndarray  # (N, 3) in deg/s, columns [x, y, z]
    fs: float  # nominal sampling rate in Hz
    source: Path

    @property
    def acc_mag(self) -> np.ndarray:
        return np.linalg.norm(self.acc, axis=1)

    @property
    def n_samples(self) -> int:
        return self.t.shape[0]


def _read_rows(path: Path) -> np.ndarray:
    """Parse the CSV, skipping rows corrupted by serial-logging glitches.

    Real recordings occasionally contain truncated or garbled lines (dropped
    bytes on the serial link), so rows that don't have exactly `_N_COLUMNS`
    numeric fields are dropped rather than failing the whole load.
    """
    rows = []
    n_skipped = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for line_no, row in enumerate(reader, start=2):
            if len(row) != _N_COLUMNS:
                n_skipped += 1
                continue
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                n_skipped += 1

    if n_skipped:
        warnings.warn(f"{path}: skipped {n_skipped} malformed row(s)")
    if not rows:
        raise ValueError(f"{path}: no valid data rows found")

    return np.array(rows)


def load_csv(path: str | Path) -> ImuRecording:
    """Load a `rec_*.csv` IMU log.

    Expected header: t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps
    """
    path = Path(path)
    data = _read_rows(path)

    t_ms = data[:, 0]
    t = (t_ms - t_ms[0]) / 1000.0

    acc = data[:, 1:4] / _MG_PER_G
    gyro = data[:, 4:7] / _MDPS_PER_DPS

    fs = 1.0 / np.median(np.diff(t))

    return ImuRecording(t=t, acc=acc, gyro=gyro, fs=fs, source=path)
