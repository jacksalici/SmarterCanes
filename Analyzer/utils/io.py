"""Loading of raw IMU CSV recordings produced by the SmartCane firmware."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Raw units used by the firmware log.
_MG_PER_G = 1000.0
_MDPS_PER_DPS = 1000.0


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


def load_csv(path: str | Path) -> ImuRecording:
    """Load a `rec_*.csv` IMU log.

    Expected header: t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps
    """
    path = Path(path)
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    t_ms = data[:, 0]
    t = (t_ms - t_ms[0]) / 1000.0

    acc = data[:, 1:4] / _MG_PER_G
    gyro = data[:, 4:7] / _MDPS_PER_DPS

    fs = 1.0 / np.median(np.diff(t))

    return ImuRecording(t=t, acc=acc, gyro=gyro, fs=fs, source=path)
