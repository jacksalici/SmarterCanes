"""Windowing preprocessing for window-based experiments.

The pipeline here turns a set of recordings into a tensor of comparable
windows, in four stages:

1. Resample onto a uniform time base. The firmware's sample interval jitters
   (the median rate across `data/` is ~67 Hz against a nominal 100 Hz), so a
   window length in seconds would otherwise map to a different number of
   samples in every recording.
2. Cut one window per anchor event (a detected step), spanning several steps.
3. Align the windows against a common template by normalized cross-correlation,
   so the same point of the gait cycle lands at the same index in every window.
   Anchoring already gets them roughly phase-locked; this removes the residual
   jitter, which a downstream model would otherwise have to spend capacity
   modelling as if it were signal.
4. Z-score each channel using statistics fitted on the training windows only.

Both the template and the normalization statistics are *fitted* on normal
training data and then applied frozen, so nothing about the held-out windows
leaks into the preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

# Channels fed to a model. dist_mm is deliberately not here by default: it is
# the ground-truth sensor for step detection, so including it would leak the
# signal a model is meant to be independent of. Callers opt in explicitly.
IMU_CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "acc_mag")
DIST_CHANNEL = "dist_mm"

# Channel used to estimate the alignment lag: acceleration magnitude, which is
# orientation-independent (the cane is not held at a repeatable angle) and is
# where the tip strike shows up most sharply.
_ALIGN_CHANNEL = "acc_mag"


@dataclass
class UniformSeries:
    """One recording resampled onto a uniform time base."""

    session_id: str
    t: np.ndarray  # (M,) uniform time base in seconds
    data: np.ndarray  # (C, M) channel values
    channels: tuple[str, ...]
    fs: float  # samples per second of this grid
    step_idx: np.ndarray  # anchor indices (detected steps) on this grid
    event_idx: np.ndarray  # user-marked sample indices on this grid
    ann: int | None


@dataclass
class NormStats:
    """Per-channel z-score statistics, fitted on training windows."""

    mean: np.ndarray  # (C,)
    std: np.ndarray  # (C,)


@dataclass
class WindowSet:
    """Aligned, normalized windows plus everything needed to interpret them."""

    x: np.ndarray  # (n_windows, C, L) normalized
    channels: tuple[str, ...]
    fs: float
    window_s: float
    session_ids: np.ndarray  # (n_windows,) session each window came from
    center_times: np.ndarray  # (n_windows,) anchor time in seconds, after alignment
    lags: np.ndarray  # (n_windows,) alignment shift applied, in samples
    is_normal: np.ndarray  # (n_windows,) bool, window treated as normal
    template: np.ndarray  # (L,) alignment template
    align_before: np.ndarray  # (n_windows, L) alignment signal at the raw anchor
    align_after: np.ndarray  # (n_windows, L) alignment signal after alignment
    norm: NormStats

    @property
    def n_windows(self) -> int:
        return self.x.shape[0]

    @property
    def window_len(self) -> int:
        return self.x.shape[2]


def to_uniform(
    rec,
    session_id: str,
    step_times: np.ndarray,
    target_fs: float,
    with_dist: bool = False,
) -> UniformSeries:
    """Resample a recording onto a uniform `target_fs` grid.

    `step_times` are the anchor times in seconds (from a step detector, run on
    the original samples); they are mapped onto the new grid by nearest
    sample. Event markers are mapped the same way rather than interpolated -
    they are impulses, and interpolation would smear or erase them.
    """
    channels = list(IMU_CHANNELS)
    if with_dist:
        if not rec.has_dist:
            raise ValueError(f"{session_id}: dist_mm requested but not present in the recording")
        channels.append(DIST_CHANNEL)

    duration = float(rec.t[-1])
    t_new = np.arange(0.0, duration, 1.0 / target_fs)

    raw = {
        "ax": rec.acc[:, 0],
        "ay": rec.acc[:, 1],
        "az": rec.acc[:, 2],
        "gx": rec.gyro[:, 0],
        "gy": rec.gyro[:, 1],
        "gz": rec.gyro[:, 2],
        "acc_mag": rec.acc_mag,
    }
    if with_dist:
        raw[DIST_CHANNEL] = rec.dist_mm

    data = np.vstack([np.interp(t_new, rec.t, raw[c]) for c in channels])

    step_idx = _nearest_indices(t_new, step_times)
    event_times = rec.t[rec.event > 0] if rec.has_event else np.empty(0)
    event_idx = _nearest_indices(t_new, event_times)

    return UniformSeries(
        session_id=session_id,
        t=t_new,
        data=data,
        channels=tuple(channels),
        fs=target_fs,
        step_idx=step_idx,
        event_idx=event_idx,
        ann=rec.ann,
    )


def _nearest_indices(t_grid: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index of the grid sample nearest each time, clipped to the grid."""
    if times.size == 0 or t_grid.size == 0:
        return np.empty(0, dtype=int)
    idx = np.searchsorted(t_grid, times)
    idx = np.clip(idx, 1, t_grid.size - 1)
    left_is_nearer = np.abs(times - t_grid[idx - 1]) <= np.abs(t_grid[idx] - times)
    return np.where(left_is_nearer, idx - 1, idx).astype(int)


def extract_anchors(
    series_list: list[UniformSeries],
    window_len: int,
    step_hop: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the anchor (series, center) pairs that yield a full window.

    A window is kept only if it fits inside its recording *with `max_lag`
    samples of slack on both sides*, so alignment can later shift it by any
    lag in range and still re-slice real samples - no zero padding, and no
    window that changes length depending on how far it moved.
    """
    half = window_len // 2
    series_i = []
    centers = []
    for i, series in enumerate(series_list):
        length = series.data.shape[1]
        for center in series.step_idx[::step_hop]:
            if center - half - max_lag < 0:
                continue
            if center - half + window_len + max_lag > length:
                continue
            series_i.append(i)
            centers.append(int(center))
    return np.array(series_i, dtype=int), np.array(centers, dtype=int)


def _slice(series: UniformSeries, center: int, window_len: int) -> np.ndarray:
    start = center - window_len // 2
    return series.data[:, start : start + window_len]


def _align_signal(series: UniformSeries, center: int, window_len: int, pad: int = 0) -> np.ndarray:
    """Mean-removed alignment channel over a window, optionally padded either side."""
    row = series.channels.index(_ALIGN_CHANNEL)
    start = center - window_len // 2 - pad
    seg = series.data[row, start : start + window_len + 2 * pad]
    return seg - seg.mean()


def _unit(v: np.ndarray) -> np.ndarray:
    """Scale to unit L2 norm, so loud windows don't dominate a mean template."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _best_lag(
    series: UniformSeries,
    center: int,
    window_len: int,
    max_lag: int,
    template: np.ndarray,
) -> int:
    """Lag in [-max_lag, max_lag] maximizing normalized cross-correlation.

    The correlation is computed with a single `correlate(..., mode="valid")`
    over an extended slice - i.e. sliding the template over the signal as a
    matched filter - which gives the numerator for every candidate lag at
    once. Each is then divided by the norm of the sub-window it came from, so
    a lag isn't preferred merely because the signal is louder there.
    """
    ext = _align_signal(series, center, window_len, pad=max_lag)
    num = correlate(ext, template, mode="valid")  # length 2*max_lag + 1

    # Per-offset norm of the mean-removed sub-window, via prefix sums.
    csum = np.concatenate([[0.0], np.cumsum(ext)])
    csum_sq = np.concatenate([[0.0], np.cumsum(ext**2)])
    n_offsets = num.size
    starts = np.arange(n_offsets)
    ends = starts + window_len
    sums = csum[ends] - csum[starts]
    sums_sq = csum_sq[ends] - csum_sq[starts]
    var = np.maximum(sums_sq - sums**2 / window_len, 0.0)
    den = np.sqrt(var) * np.linalg.norm(template)

    ncc = np.where(den > 0, num / np.where(den > 0, den, 1.0), -np.inf)
    return int(np.argmax(ncc)) - max_lag


def align_windows(
    series_list: list[UniformSeries],
    series_i: np.ndarray,
    centers: np.ndarray,
    window_len: int,
    max_lag: int,
    fit_mask: np.ndarray,
    n_iters: int = 3,
    template: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a per-window alignment lag against a common template.

    The template is refined iteratively on the `fit_mask` windows only: start
    from their mean alignment signal at the raw anchors, re-estimate every
    lag against it, rebuild the template from the shifted windows, repeat.
    Iteration is what breaks the chicken-and-egg between "where is the
    pattern" and "where is each window relative to it"; it converges in a
    couple of passes because step anchoring already puts windows within a
    fraction of a cycle of each other. Pass a `template` to skip fitting and
    align against a frozen one (as when scoring with a saved model).

    Returns (lags, template).
    """
    fit_idx = np.flatnonzero(fit_mask)
    if fit_idx.size == 0:
        raise ValueError("no windows available to fit the alignment template")

    def signal_at(i: int, lag: int) -> np.ndarray:
        return _align_signal(series_list[series_i[i]], centers[i] + lag, window_len)

    lags = np.zeros(centers.size, dtype=int)

    if template is None:
        template = np.mean([_unit(signal_at(i, 0)) for i in fit_idx], axis=0)

        for _ in range(n_iters):
            new_lags = np.array(
                [
                    _best_lag(
                        series_list[series_i[i]], centers[i], window_len, max_lag, template
                    )
                    for i in fit_idx
                ]
            )
            shift = np.mean(np.abs(new_lags - lags[fit_idx]))
            lags[fit_idx] = new_lags
            template = np.mean([_unit(signal_at(i, lags[i])) for i in fit_idx], axis=0)
            if shift < 1.0:
                break

    # Final pass over every window (fit or held out) against the frozen template.
    for i in range(centers.size):
        lags[i] = _best_lag(
            series_list[series_i[i]], centers[i], window_len, max_lag, template
        )

    return lags, template


def fit_norm_stats(x: np.ndarray) -> NormStats:
    """Per-channel mean/std over a window tensor of shape (n, C, L).

    Statistics are global over the training windows rather than per-window:
    normalizing each window on its own would erase amplitude differences,
    which for anomaly detection are exactly the thing worth noticing.
    """
    mean = x.mean(axis=(0, 2))
    std = x.std(axis=(0, 2))
    std = np.where(std > 0, std, 1.0)
    return NormStats(mean=mean, std=std)


def apply_norm(x: np.ndarray, stats: NormStats) -> np.ndarray:
    return (x - stats.mean[None, :, None]) / stats.std[None, :, None]


def build_window_set(
    series_list: list[UniformSeries],
    window_s: float,
    step_hop: int = 1,
    max_lag_s: float = 1.0,
    align_iters: int = 3,
    normal_mask_fn=None,
    template: np.ndarray | None = None,
    norm: NormStats | None = None,
) -> WindowSet:
    """Run the whole pipeline: cut, align, mark normal/held-out, normalize.

    `normal_mask_fn(series, center, window_len)` decides whether a window
    counts as normal training data; it defaults to all-normal. `template` and
    `norm` may be supplied to reuse a fitted preprocessing state instead of
    refitting (scoring with a saved model).
    """
    if not series_list:
        raise ValueError("no recordings to window")

    fs = series_list[0].fs
    window_len = int(round(window_s * fs))
    max_lag = int(round(max_lag_s * fs))

    series_i, centers = extract_anchors(series_list, window_len, step_hop, max_lag)
    if centers.size == 0:
        raise ValueError(
            f"no windows of {window_s:g} s fit inside any recording - "
            "try a shorter --window-s or a smaller --max-lag-s"
        )

    if normal_mask_fn is None:
        is_normal = np.ones(centers.size, dtype=bool)
    else:
        is_normal = np.array(
            [
                bool(normal_mask_fn(series_list[series_i[i]], int(centers[i]), window_len))
                for i in range(centers.size)
            ]
        )

    align_before = np.stack(
        [
            _align_signal(series_list[series_i[i]], int(centers[i]), window_len)
            for i in range(centers.size)
        ]
    )

    # The template is fitted on normal windows only - unless there are none,
    # in which case every window is a candidate and there is nothing to hold
    # back anyway.
    fit_mask = is_normal if is_normal.any() else np.ones_like(is_normal)
    lags, template = align_windows(
        series_list,
        series_i,
        centers,
        window_len,
        max_lag,
        fit_mask=fit_mask,
        n_iters=align_iters,
        template=template,
    )

    aligned_centers = centers + lags
    x = np.stack(
        [
            _slice(series_list[series_i[i]], int(aligned_centers[i]), window_len)
            for i in range(centers.size)
        ]
    )
    align_after = np.stack(
        [
            _align_signal(series_list[series_i[i]], int(aligned_centers[i]), window_len)
            for i in range(centers.size)
        ]
    )

    if norm is None:
        norm = fit_norm_stats(x[fit_mask])

    center_times = np.array(
        [series_list[series_i[i]].t[int(aligned_centers[i])] for i in range(centers.size)]
    )
    session_ids = np.array([series_list[i].session_id for i in series_i])

    return WindowSet(
        x=apply_norm(x, norm),
        channels=series_list[0].channels,
        fs=fs,
        window_s=window_s,
        session_ids=session_ids,
        center_times=center_times,
        lags=lags,
        is_normal=is_normal,
        template=template,
        align_before=align_before,
        align_after=align_after,
        norm=norm,
    )
