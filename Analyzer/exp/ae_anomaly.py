"""Experiment 3: evaluate the autoencoder as an anomaly detector, train vs test.

Fits everything - alignment template, normalization statistics, weights,
decision threshold - on the `train` split (normal gait only), then scores the
`test` split without refitting anything.

Labels come from the filename annotation:

- `ann0`  normal, every window is a negative
- `ann1`  abnormal throughout, every window is a positive
- `ann-1` normal except at least one marked moment (the in-recording `event`
          click); only windows overlapping a marker are positives, the rest
          are dropped rather than called normal

Default windowing is `anchor="slide"` (a fixed time grid) rather than one
window per detected step, since every step detector here degrades on
abnormal gait and would otherwise decide how much evidence a recording
yields. Default alignment avoids correlating against the normal template,
since that lets an abnormal window search for the shift that looks most
normal and shrinks its own error.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from exp.step_ae import StepAEConfig, _train, build_model, resolve_with_dist
from exp.step_count import count_steps
from utils.io import group_sessions, load_session
from utils.metrics import (
    ThresholdMetrics,
    average_precision,
    best_f1,
    prevalence,
    roc_auc,
    threshold_metrics,
)
from utils.io import DIST_ERROR_FLOOR_MM
from utils.windows import (
    DIST_CHANNEL,
    UniformSeries,
    WindowSet,
    build_window_set,
    to_uniform,
)

# Worst fraction of time samples the "peak" score_agg averages over.
_PEAK_FRACTION = 0.1

# Labels carried per window. DROP means "not scoreable", not "normal".
LABEL_NORMAL = 0
LABEL_ANOMALY = 1
LABEL_DROP = -1


@dataclass
class SessionReport:
    """How the detector did on one test session."""

    session_id: str
    ann: int | None
    n_windows: int  # windows cut from this session
    n_labeled: int  # of those, the ones carrying a usable label
    label: int  # the label those windows carry
    n_flagged: int  # labeled windows scoring above the threshold
    flag_rate: float
    median_score: float
    max_score: float
    description: str = ""
    dist_clipped_frac: float | None = None  # fraction of dist_mm clamped to the error floor

    @property
    def detected(self) -> bool:
        """Majority of its scoreable windows were flagged."""
        return self.flag_rate >= 0.5

    @property
    def correct(self) -> bool:
        return self.detected == (self.label == LABEL_ANOMALY)


@dataclass
class AEAnomalyResult:
    config: StepAEConfig
    train_windows: WindowSet
    test_windows: WindowSet
    train_split: np.ndarray  # (n_train,) "train" | "val"
    train_loss: np.ndarray
    val_loss: np.ndarray
    best_epoch: int
    val_scores: np.ndarray  # calibration scores, held-out normal training windows
    train_scores: np.ndarray
    scores: np.ndarray  # (n_test,) anomaly score per test window
    labels: np.ndarray  # (n_test,) LABEL_*
    recon: np.ndarray  # (n_test, C, L)
    channel_error: np.ndarray  # (n_test, C)
    threshold: float
    sessions: list[SessionReport]
    notes: list[str] = field(default_factory=list)

    @property
    def scoreable(self) -> np.ndarray:
        return self.labels != LABEL_DROP

    @property
    def y_true(self) -> np.ndarray:
        return self.labels[self.scoreable]

    @property
    def y_score(self) -> np.ndarray:
        return self.scores[self.scoreable]

    @property
    def window_metrics(self) -> ThresholdMetrics:
        return threshold_metrics(self.y_true, self.y_score, self.threshold)

    @property
    def oracle_metrics(self) -> ThresholdMetrics:
        return best_f1(self.y_true, self.y_score)

    @property
    def auprc(self) -> float:
        return average_precision(self.y_true, self.y_score)

    @property
    def roc_auc(self) -> float:
        return roc_auc(self.y_true, self.y_score)

    @property
    def prevalence(self) -> float:
        return prevalence(self.y_true)

    @property
    def session_accuracy(self) -> float:
        if not self.sessions:
            return float("nan")
        return sum(s.correct for s in self.sessions) / len(self.sessions)


def score_windows(
    model, x: np.ndarray, config: StepAEConfig, chan_scale: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct `x` and collapse the error into one anomaly score per window.

    - `mean` - plain mean squared error.
    - `chan_norm` - each channel's error divided by its median on held-out
      normal windows (`chan_scale`) before averaging, so no channel's natural
      residual dominates regardless of whether it carries the anomaly.
    - `peak` - mean over the worst `_PEAK_FRACTION` of time samples.
    """
    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(x).float()).numpy()
    squared_error = (recon - x) ** 2  # (N, C, L)
    channel_error = squared_error.mean(axis=2)  # (N, C)

    if config.score_agg == "mean":
        scores = squared_error.mean(axis=(1, 2))
    elif config.score_agg == "chan_norm":
        if chan_scale is None:
            raise ValueError("score_agg='chan_norm' needs chan_scale fitted on normal windows")
        scores = (channel_error / chan_scale[None, :]).mean(axis=1)
    elif config.score_agg == "peak":
        per_sample = squared_error.mean(axis=1)  # (N, L), across channels
        k = max(1, int(round(_PEAK_FRACTION * per_sample.shape[1])))
        scores = np.sort(per_sample, axis=1)[:, -k:].mean(axis=1)
    else:
        raise ValueError(f"unknown score_agg: {config.score_agg!r}")

    return scores, channel_error, recon


def load_descriptions(path: Path) -> dict[str, str]:
    """Read the semicolon-separated anomaly descriptions keyed by recording
    number. Purely for reporting; a missing file just yields no descriptions.
    """
    descriptions: dict[str, str] = {}
    if not path.exists():
        return descriptions
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        descriptions[f"{int(parts[0]):05d}"] = parts[1]
    return descriptions


def load_series(paths: list[Path], config: StepAEConfig) -> list[UniformSeries]:
    """Load every session among `paths` and resample onto the uniform grid.

    A session with no detected steps is dropped only under `anchor="step"`,
    which needs them; sliding anchors keep it.
    """
    sessions = group_sessions(sorted(paths))
    if not sessions:
        raise ValueError("no rec_*.csv recordings found")

    series_list = []
    for session_id, paths in sorted(sessions.items()):
        rec = load_session(paths)
        step_times = count_steps(rec).step_times
        if step_times.size == 0 and config.anchor == "step":
            warnings.warn(f"rec_{session_id}: no steps detected, skipping")
            continue
        series_list.append(
            to_uniform(
                rec,
                session_id=session_id,
                step_times=step_times,
                target_fs=config.target_fs,
                with_dist=config.with_dist,
            )
        )
    return series_list


def _dist_clipped_frac(series: UniformSeries) -> float | None:
    """Fraction of this session's distance trace clamped to the error floor
    by `to_uniform` (sensor dropout, not signal).
    """
    if DIST_CHANNEL not in series.channels:
        return None
    row = series.channels.index(DIST_CHANNEL)
    return float(np.mean(series.data[row] <= DIST_ERROR_FLOOR_MM))


def label_windows(windows: WindowSet, series_list: list[UniformSeries]) -> np.ndarray:
    """Assign LABEL_* per window from the session annotation and event markers,
    judged on the aligned center - what the model actually saw.
    """
    by_session = {s.session_id: s for s in series_list}
    half = windows.window_s / 2.0

    labels = np.full(windows.n_windows, LABEL_DROP, dtype=int)
    for i, session_id in enumerate(windows.session_ids):
        series = by_session[str(session_id)]
        if series.ann == 0:
            labels[i] = LABEL_NORMAL
        elif series.ann == 1:
            labels[i] = LABEL_ANOMALY
        elif series.ann == -1:
            start, end = windows.center_times[i] - half, windows.center_times[i] + half
            event_times = series.t[series.event_idx] if series.event_idx.size else np.empty(0)
            if np.any((event_times >= start) & (event_times <= end)):
                labels[i] = LABEL_ANOMALY
    return labels


def split_train_val(
    windows: WindowSet, val_frac: float, window_s: float
) -> tuple[np.ndarray, list[str]]:
    """Hold out the last `val_frac` of each training session, in time - not a
    random window-level split (neighbouring windows overlap heavily) or a
    whole-session holdout (too few sessions). A one-window guard band at the
    cut keeps train and val from sharing a sample.
    """
    split = np.full(windows.n_windows, "train", dtype=object)
    notes: list[str] = []

    for session_id in np.unique(windows.session_ids):
        idx = np.flatnonzero(windows.session_ids == session_id)
        order = idx[np.argsort(windows.center_times[idx])]
        n_val = max(1, int(round(val_frac * order.size)))
        val_idx = order[-n_val:]
        split[val_idx] = "val"

        cut_time = windows.center_times[val_idx].min()
        guard = idx[
            (windows.center_times[idx] < cut_time)
            & (windows.center_times[idx] > cut_time - window_s)
        ]
        split[guard] = "guard"
        notes.append(
            f"rec_{session_id}: {order.size - n_val - guard.size} train / {n_val} val "
            f"(+{guard.size} dropped as guard band)"
        )

    return split.astype(str), notes


def run_ae_anomaly(
    train_paths: list[Path],
    test_paths: list[Path],
    config: StepAEConfig,
    descriptions: dict[str, str] | None = None,
) -> AEAnomalyResult:
    """Fit on `train_paths`, score `test_paths`, and measure."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    descriptions = descriptions or {}

    # Resolved across both splits: dist_mm is only usable if every recording
    # on both sides carries it.
    config, dist_notes = resolve_with_dist(config, sorted(train_paths) + sorted(test_paths))

    train_series = load_series(train_paths, config)
    train_windows = build_window_set(
        train_series,
        window_s=config.window_s,
        step_hop=config.step_hop,
        max_lag_s=config.max_lag_s,
        align_iters=config.align_iters,
        anchor=config.anchor,
        hop_s=config.hop_s,
        align_mode=config.align_mode,
    )
    split, notes = split_train_val(train_windows, config.val_frac, config.window_s)

    model = build_model(config, train_windows.x.shape[1], train_windows.x.shape[2])
    x_train = torch.from_numpy(train_windows.x[split == "train"]).float()
    x_val = torch.from_numpy(train_windows.x[split == "val"]).float()
    train_loss, val_loss, best_epoch = _train(model, x_train, x_val, config)

    chan_scale = None
    if config.score_agg == "chan_norm":
        _, val_channel_error, _ = score_windows(
            model, train_windows.x[split == "val"], replace(config, score_agg="mean")
        )
        chan_scale = np.maximum(np.median(val_channel_error, axis=0), 1e-12)

    train_scores, _, _ = score_windows(
        model, train_windows.x[split == "train"], config, chan_scale
    )
    val_scores, _, _ = score_windows(model, train_windows.x[split == "val"], config, chan_scale)

    threshold = float(np.percentile(val_scores, config.threshold_pct))

    test_series = load_series(test_paths, config)
    test_windows = build_window_set(
        test_series,
        window_s=config.window_s,
        step_hop=config.step_hop,
        max_lag_s=config.max_lag_s,
        align_iters=config.align_iters,
        anchor=config.anchor,
        hop_s=config.hop_s,
        align_mode=config.align_mode,
        template=train_windows.template,
        norm=train_windows.norm,
    )
    scores, channel_error, recon = score_windows(model, test_windows.x, config, chan_scale)
    labels = label_windows(test_windows, test_series)

    sessions = []
    for series in test_series:
        mask = test_windows.session_ids == series.session_id
        labeled = mask & (labels != LABEL_DROP)
        n_labeled = int(labeled.sum())
        session_scores = scores[labeled]
        sessions.append(
            SessionReport(
                session_id=series.session_id,
                ann=series.ann,
                n_windows=int(mask.sum()),
                n_labeled=n_labeled,
                label=int(labels[labeled][0]) if n_labeled else LABEL_DROP,
                n_flagged=int(np.sum(session_scores > threshold)),
                flag_rate=float(np.mean(session_scores > threshold)) if n_labeled else float("nan"),
                median_score=float(np.median(session_scores)) if n_labeled else float("nan"),
                max_score=float(session_scores.max()) if n_labeled else float("nan"),
                description=descriptions.get(series.session_id, ""),
                dist_clipped_frac=_dist_clipped_frac(series),
            )
        )
    sessions = [s for s in sessions if s.n_labeled > 0]

    return AEAnomalyResult(
        config=config,
        train_windows=train_windows,
        test_windows=test_windows,
        train_split=split,
        train_loss=train_loss,
        val_loss=val_loss,
        best_epoch=best_epoch,
        val_scores=val_scores,
        train_scores=train_scores,
        scores=scores,
        labels=labels,
        recon=recon,
        channel_error=channel_error,
        threshold=threshold,
        sessions=sessions,
        notes=dist_notes + notes,
    )
