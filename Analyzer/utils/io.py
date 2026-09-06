"""Loading of raw IMU CSV recordings produced by the SmartCane firmware."""

from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Raw units used by the firmware log.
_MG_PER_G = 1000.0
_MDPS_PER_DPS = 1000.0

_BASE_COLUMNS = 7  # t_ms, ax_mg, ay_mg, az_mg, gx_mdps, gy_mdps, gz_mdps
_DIST_COLUMN_NAME = "dist_mm"  # optional 8th column: cane base to ground, in mm

# rec_00001.csv (legacy, single file) / rec_00001_seg002.csv (segmented) /
# either with a trailing _ann{-1,0,1} stop annotation.
_SESSION_FILENAME_RE = re.compile(r"^rec_(?P<session>\d+)(?:_seg(?P<segment>\d+))?(?:_ann-?\d+)?$")


@dataclass
class ImuRecording:
    """A single IMU recording, converted to physical units."""

    t: np.ndarray  # seconds, starting at 0
    acc: np.ndarray  # (N, 3) in g, columns [x, y, z]
    gyro: np.ndarray  # (N, 3) in deg/s, columns [x, y, z]
    fs: float  # nominal sampling rate in Hz
    source: Path
    dist_mm: np.ndarray | None = None  # cane base-to-ground distance, if logged

    @property
    def acc_mag(self) -> np.ndarray:
        return np.linalg.norm(self.acc, axis=1)

    @property
    def n_samples(self) -> int:
        return self.t.shape[0]

    @property
    def has_dist(self) -> bool:
        return self.dist_mm is not None


def _read_rows(path: Path, n_columns: int) -> np.ndarray:
    """Parse the CSV, skipping rows corrupted by serial-logging glitches.

    Real recordings occasionally contain truncated or garbled lines (dropped
    bytes on the serial link), so rows that don't have exactly `n_columns`
    numeric fields are dropped rather than failing the whole load.
    """
    rows = []
    n_skipped = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for line_no, row in enumerate(reader, start=2):
            if len(row) != n_columns:
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
    optionally followed by dist_mm (cane base to ground, from a distance
    sensor), which recordings from newer firmware include.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        header = next(csv.reader(f))

    has_dist = len(header) > _BASE_COLUMNS and header[_BASE_COLUMNS].strip() == _DIST_COLUMN_NAME
    n_columns = _BASE_COLUMNS + 1 if has_dist else _BASE_COLUMNS

    data = _read_rows(path, n_columns)

    t_ms = data[:, 0]
    t = (t_ms - t_ms[0]) / 1000.0

    acc = data[:, 1:4] / _MG_PER_G
    gyro = data[:, 4:7] / _MDPS_PER_DPS
    dist_mm = data[:, _BASE_COLUMNS] if has_dist else None

    fs = 1.0 / np.median(np.diff(t))

    return ImuRecording(t=t, acc=acc, gyro=gyro, fs=fs, source=path, dist_mm=dist_mm)


def group_sessions(paths: list[Path]) -> dict[str, list[Path]]:
    """Group recording files by session, in segment order.

    Newer firmware splits a session into 30-second segment files
    (`rec_XXXXX_segNNN...`) instead of one continuous one, so a session's
    data has to be read back as the concatenation of its segments, in
    order. A legacy single-file session (`rec_XXXXX_annY.csv`, no `_seg`)
    just forms a group of one.
    """
    groups: dict[str, list[Path]] = {}
    for path in paths:
        m = _SESSION_FILENAME_RE.match(path.stem)
        if not m:
            continue
        groups.setdefault(m.group("session"), []).append(path)

    def segment_number(path: Path) -> int:
        m = _SESSION_FILENAME_RE.match(path.stem)
        segment = m.group("segment") if m else None
        return int(segment) if segment is not None else 0

    for paths_in_session in groups.values():
        paths_in_session.sort(key=segment_number)

    return groups


def load_session(paths: list[Path]) -> ImuRecording:
    """Load a session's segment file(s) as one continuous recording.

    `paths` must be a single session's segments, already in order (as
    `group_sessions` produces). Each segment's `t_ms` restarts from 0, so
    segments are stitched back-to-back by offsetting each one to start right
    after the previous segment ends.
    """
    if len(paths) == 1:
        return load_csv(paths[0])

    segments = [load_csv(p) for p in paths]

    t_parts = []
    offset = 0.0
    for seg in segments:
        t_parts.append(seg.t + offset)
        step = np.median(np.diff(seg.t)) if seg.t.size > 1 else 0.0
        offset += seg.t[-1] + step if seg.t.size else 0.0

    t = np.concatenate(t_parts)
    acc = np.concatenate([seg.acc for seg in segments])
    gyro = np.concatenate([seg.gyro for seg in segments])

    dist_mm = None
    if all(seg.has_dist for seg in segments):
        dist_mm = np.concatenate([seg.dist_mm for seg in segments])

    fs = 1.0 / np.median(np.diff(t))

    return ImuRecording(t=t, acc=acc, gyro=gyro, fs=fs, source=paths[0], dist_mm=dist_mm)
