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

# Columns every recording has, in firmware write order.
_REQUIRED_COLUMNS = ("t_ms", "ax_mg", "ay_mg", "az_mg", "gx_mdps", "gy_mdps", "gz_mdps")
# Columns added by later firmware revisions. They are resolved by *name*, not
# position, so a recording may carry either, both or neither: `dist_mm` is the
# cane base to ground distance, `event` marks the first sample after a single
# click during recording (a user-flagged moment of interest).
_DIST_COLUMN = "dist_mm"
_EVENT_COLUMN = "event"

# Bands of the Modulino Distance sensor reading, in millimetres. The reading
# is between the cane base and the ground and quantizes into three bands rather
# than behaving like a clean waveform: 90-110 mm is the resting/noise band
# (cane planted, sensor jitter only), above 118 mm the tip has actually lifted
# away from the ground, and below 90 mm is a measurement error rather than a
# real reading. Empirical calibration values for this sensor and mounting.
# They live here rather than in a single experiment because both the step
# detector and the windowing preprocessor have to know where the error floor is.
DIST_ERROR_FLOOR_MM = 90.0
DIST_REST_HIGH_MM = 110.0
DIST_STEP_THRESHOLD_MM = 118.0

# rec_00001.csv (legacy, single file) / rec_00001_seg002.csv (segmented) /
# either with a trailing _ann{-1,0,1} stop annotation.
_SESSION_FILENAME_RE = re.compile(
    r"^rec_(?P<session>\d+)(?:_seg(?P<segment>\d+))?(?:_ann(?P<ann>-?\d+))?$"
)


@dataclass
class ImuRecording:
    """A single IMU recording, converted to physical units."""

    t: np.ndarray  # seconds, starting at 0
    acc: np.ndarray  # (N, 3) in g, columns [x, y, z]
    gyro: np.ndarray  # (N, 3) in deg/s, columns [x, y, z]
    fs: float  # nominal sampling rate in Hz
    source: Path
    dist_mm: np.ndarray | None = None  # cane base-to-ground distance, if logged
    event: np.ndarray | None = None  # 1 on user-marked samples, if logged
    ann: int | None = None  # stop annotation from the filename, if present

    @property
    def acc_mag(self) -> np.ndarray:
        return np.linalg.norm(self.acc, axis=1)

    @property
    def n_samples(self) -> int:
        return self.t.shape[0]

    @property
    def has_dist(self) -> bool:
        return self.dist_mm is not None

    @property
    def has_event(self) -> bool:
        return self.event is not None


def parse_ann(path: str | Path) -> int | None:
    """Read the stop annotation encoded in a recording's filename.

    Every segment of a session carries the same `_annY` suffix, so the label
    can be recovered from any one of its files. Returns None for the legacy
    filenames that have no annotation.
    """
    m = _SESSION_FILENAME_RE.match(Path(path).stem)
    if not m or m.group("ann") is None:
        return None
    return int(m.group("ann"))


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


def _column_indices(path: Path, header: list[str]) -> dict[str, int]:
    """Map column name to column index, validating the required columns.

    Resolving by name (rather than the fixed positions the firmware happens
    to write) is what lets one loader read every schema revision: columns
    appended by newer firmware simply show up in the map.
    """
    indices = {name.strip(): i for i, name in enumerate(header)}
    missing = [name for name in _REQUIRED_COLUMNS if name not in indices]
    if missing:
        raise ValueError(f"{path}: missing required column(s): {', '.join(missing)}")
    return indices


def load_csv(path: str | Path) -> ImuRecording:
    """Load a `rec_*.csv` IMU log.

    Required header columns: t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps.
    Newer firmware appends dist_mm (cane base to ground, from a distance
    sensor) and event (user-marked sample); both are optional.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        header = next(csv.reader(f))

    indices = _column_indices(path, header)
    data = _read_rows(path, len(header))

    t_ms = data[:, indices["t_ms"]]
    t = (t_ms - t_ms[0]) / 1000.0

    acc = np.column_stack([data[:, indices[c]] for c in ("ax_mg", "ay_mg", "az_mg")]) / _MG_PER_G
    gyro = (
        np.column_stack([data[:, indices[c]] for c in ("gx_mdps", "gy_mdps", "gz_mdps")])
        / _MDPS_PER_DPS
    )
    dist_mm = data[:, indices[_DIST_COLUMN]] if _DIST_COLUMN in indices else None
    event = data[:, indices[_EVENT_COLUMN]] if _EVENT_COLUMN in indices else None

    fs = 1.0 / np.median(np.diff(t))

    return ImuRecording(
        t=t,
        acc=acc,
        gyro=gyro,
        fs=fs,
        source=path,
        dist_mm=dist_mm,
        event=event,
        ann=parse_ann(path),
    )


def has_dist_column(path: str | Path) -> bool:
    """Whether a recording carries `dist_mm`, read from the header alone.

    Only the first line is parsed, so this is cheap enough to run over a whole
    directory before deciding whether the distance channel is available - which
    is what lets `--with-dist` default to "use it when every recording has it"
    rather than failing on the older firmware revisions that predate the sensor.
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        header = next(csv.reader(f))
    return _DIST_COLUMN in {name.strip() for name in header}


def load_split(path: str | Path) -> dict[str, str]:
    """Read the `filename,split` table that assigns each recording to
    `train`, `test` or `walk`.

    `train` is the normal-only pool `ae-anomaly` fits on; `test` is the
    labelled pool it evaluates against, normal and anomalous alike; `walk` is
    the rest - ordinary walking that plays no part in that protocol.
    """
    path = Path(path)
    split: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split[row["filename"]] = row["split"]
    return split


def paths_for_split(data_dir: str | Path, split_csv: str | Path, splits: set[str]) -> list[Path]:
    """List the recordings in `data_dir` whose split falls in `splits`."""
    data_dir = Path(data_dir)
    split = load_split(split_csv)
    return sorted(
        data_dir / name for name, s in split.items() if s in splits and (data_dir / name).exists()
    )


def normal_paths(data_dir: str | Path) -> list[Path]:
    """List every `ann0` (or legacy, un-annotated) recording in `data_dir`.

    This is the pool ordinary gait analysis - `step-count`, `step-accuracy`,
    `step-ae` - runs over by default: every recording not carrying an
    annotated abnormal stretch, `train` and `walk` alike, plus any `test`
    recording (such as the held-out normal control session) that happens to
    be `ann0` too. `ann1`/`ann-1` recordings are deliberately excluded - they
    exist to be scored by `ae-anomaly`, not averaged into a step-count report.
    """
    data_dir = Path(data_dir)
    return sorted(p for p in data_dir.glob("*.csv") if parse_ann(p) in (None, 0))


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

    # An optional column is only usable for the session as a whole if every
    # segment carries it - a half-populated array would silently misalign.
    dist_mm = None
    if all(seg.has_dist for seg in segments):
        dist_mm = np.concatenate([seg.dist_mm for seg in segments])

    event = None
    if all(seg.has_event for seg in segments):
        event = np.concatenate([seg.event for seg in segments])

    fs = 1.0 / np.median(np.diff(t))

    return ImuRecording(
        t=t,
        acc=acc,
        gyro=gyro,
        fs=fs,
        source=paths[0],
        dist_mm=dist_mm,
        event=event,
        ann=parse_ann(paths[0]),
    )
