"""Experiment 3: evaluate the autoencoder as an anomaly detector, train vs test.

`step-ae` trains and scores one pool of recordings. This runs the protocol an
actual detector has to survive: fit everything - alignment template,
normalization statistics, weights, decision threshold - on `data/train`, which
is normal gait only, then score `data/test` without refitting anything, and
measure against labels the model never saw.

Labels come from the filename annotation, with one exception:

- `ann0`  normal, every window is a negative
- `ann1`  the whole recording is abnormal gait, every window is a positive
- `ann-1` a normal recording containing at least one abnormal moment, marked
          by the in-recording `event` click. Only windows overlapping a marker
          are positives; the rest are *dropped* rather than called normal,
          because nothing says where the abnormal stretch ends.

Two failure modes of the original pipeline are worth stating, since the
protocol here is built around them.

The first is windowing. Anchoring a window on each detected step makes the
amount of evidence gathered from a recording depend on a step detector, and
every step detector available here degrades on abnormal gait - a shuffled or
dragged step produces neither a clean acceleration shock nor a tip lift past
the `dist_mm` threshold. Step anchoring yields one window each from rec_00018
and rec_00027 against 121 from the normal rec_00017, so the recordings that
matter most are the ones least measured. Sliding windows on a fixed time grid
(`anchor="slide"`) give every recording the same coverage per second.

The second is the alignment stage. Cross-correlating each window against the
*normal* template and keeping the best-matching shift lets an abnormal window
search for the pose that looks most normal, shrinking the reconstruction error
the score is made of. A convolutional autoencoder is shift-tolerant by
construction, so it can take unaligned windows and leave that error intact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from exp.step_ae import StepAEConfig, _train, build_model
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

# Fraction of a window the "peak" aggregation averages over: the worst 10% of
# time samples. A fall or a stumble occupies a fraction of a six-second window,
# so averaging over the whole window dilutes it with the ordinary walking
# either side.
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
    # Fraction of this session's dist_mm samples that were sensor dropout and
    # got clamped to the error floor. Only meaningful under --with-dist, and
    # worth carrying because it is the one way the distance channel could be
    # helping for the wrong reason.
    dist_clipped_frac: float | None = None

    @property
    def detected(self) -> bool:
        """Session-level verdict: most of its scoreable windows were flagged.

        Majority rather than "any window flagged": on a 20-30 s recording of
        continuously abnormal gait, a detector that fires once and stays quiet
        is not the same as one that recognises the gait, and "any" would call
        every long recording anomalous on its worst second alone.
        """
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

    Three ways to collapse it, because the mean over every channel and sample
    is not obviously the right one:

    - `mean` - the plain mean squared error.
    - `chan_norm` - each channel's error divided by its median on held-out
      normal windows before averaging. Without it the channels with the
      largest natural residual dominate the score whether or not they carry
      the anomaly; with it, every channel reports in units of "how unusual is
      this for this channel".
    - `peak` - the mean over the worst tenth of time samples. A stumble or a
      fall occupies a fraction of a window, and averaging across the whole
      window dilutes it with the ordinary walking on either side.

    `chan_scale` is fitted on normal training windows and passed in frozen.
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
    """Read the semicolon-separated anomaly descriptions keyed by recording number.

    Purely for reporting - nothing downstream branches on the text. Missing or
    unreadable file just means the reports carry no description column.
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


def load_series(data_dir: Path, config: StepAEConfig) -> list[UniformSeries]:
    """Load every session in a directory and resample onto the uniform grid.

    Step times are still computed, because `anchor="step"` needs them and they
    cost little; with `anchor="slide"` nothing downstream reads them. A session
    with no detected steps is therefore kept under sliding anchors - dropping
    it would be the step detector silently deciding what gets evaluated.
    """
    sessions = group_sessions(sorted(data_dir.glob("*.csv")))
    if not sessions:
        raise ValueError(f"no rec_*.csv recordings found in {data_dir}")

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
    """How much of this session's distance trace was sensor dropout.

    Readings under the error floor are clamped by `to_uniform`, so they survive
    as an exact run at the floor value. The number matters for interpretation:
    if dropout clustered in the abnormal recordings, a gain from the distance
    channel could be the model spotting a failing sensor rather than a failing
    gait.
    """
    if DIST_CHANNEL not in series.channels:
        return None
    row = series.channels.index(DIST_CHANNEL)
    return float(np.mean(series.data[row] <= DIST_ERROR_FLOOR_MM))


def label_windows(windows: WindowSet, series_list: list[UniformSeries]) -> np.ndarray:
    """Assign LABEL_* per window from the session annotation and event markers.

    Labelling happens on the *aligned* center, so a window is judged by the
    stretch of signal the model actually saw rather than by where it was cut.
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
            # else: left as LABEL_DROP - unmarked stretches of an ann-1
            # recording are unlabelled, not known-normal.
    return labels


def split_train_val(
    windows: WindowSet, val_frac: float, window_s: float
) -> tuple[np.ndarray, list[str]]:
    """Hold out the last `val_frac` of each training session, in time.

    Not a random window-level split: neighbouring windows overlap heavily, so
    random assignment puts near-duplicates on both sides and the validation
    loss stops measuring generalization. Not a whole-session holdout either -
    there are only two training sessions, so holding one out would calibrate
    the threshold on a single walk.

    A guard band of one full window on each side of the cut is discarded, so
    no training window shares a single sample with a validation window.
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


# The pipeline variants the ablation walks, in the order they were arrived at.
# Each row changes one thing from the row above it. Every field a variant
# varies is set explicitly in every variant, so the comparison does not quietly
# inherit whatever the caller happened to configure - running the ablation with
# --with-dist would otherwise hand the distance channel to all seven rows and
# flatten the very comparison the last row exists to make.
ABLATION_VARIANTS: tuple[tuple[str, dict], ...] = (
    ("step anchor + xcorr align + MLP (original)",
     dict(anchor="step", align_mode="xcorr", model="mlp", with_dist=False)),
    ("slide anchor + xcorr align + MLP",
     dict(anchor="slide", align_mode="xcorr", model="mlp", with_dist=False)),
    ("slide anchor + xcorr align + conv",
     dict(anchor="slide", align_mode="xcorr", model="conv", with_dist=False)),
    ("slide anchor + no align + conv",
     dict(anchor="slide", align_mode="none", model="conv", with_dist=False)),
    ("slide anchor + step align + conv",
     dict(anchor="slide", align_mode="step", model="conv", with_dist=False)),
    ("slide anchor + mixed align + conv",
     dict(anchor="slide", align_mode="mixed", model="conv", with_dist=False)),
    ("slide anchor + mixed align + conv + dist",
     dict(anchor="slide", align_mode="mixed", model="conv", with_dist=True)),
)


def run_ablation(
    train_dir: Path,
    test_dir: Path,
    config: StepAEConfig,
    descriptions: dict[str, str] | None = None,
    variants: tuple[tuple[str, dict], ...] = ABLATION_VARIANTS,
    seeds: tuple[int, ...] = (0,),
) -> list[tuple[str, list[AEAnomalyResult]]]:
    """Re-run the evaluation across pipeline variants, holding everything else fixed.

    Worth the extra minutes because the two headline changes - sliding anchors
    and dropping alignment - are justified by an argument about how the
    pipeline can fail, and an argument like that should be checked rather than
    asserted. Note that the variants are compared on the same test set the
    headline numbers come from, so this is a sensitivity check, not an
    independent model selection.

    Pass several `seeds` when variants come out close: the gaps between the
    better ones here are a few thousandths of AUPRC, which is the same order as
    the spread across random initializations, and a single run cannot tell
    those apart.
    """
    results = []
    for name, overrides in variants:
        runs = [
            run_ae_anomaly(train_dir, test_dir, replace(config, **overrides, seed=seed),
                           descriptions)
            for seed in seeds
        ]
        results.append((name, runs))
    return results


def run_ae_anomaly(
    train_dir: Path,
    test_dir: Path,
    config: StepAEConfig,
    descriptions: dict[str, str] | None = None,
) -> AEAnomalyResult:
    """Fit on `train_dir`, score `test_dir`, and measure."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    descriptions = descriptions or {}

    # --- fit: preprocessing and model, on normal training data only ---
    train_series = load_series(train_dir, config)
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

    # The per-channel scale for `chan_norm` is fitted on the held-out normal
    # windows, then frozen - like the threshold, it never sees a test label.
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

    # Threshold from held-out normal windows only: no test label is consulted,
    # so this is an operating point a deployed detector could actually set.
    threshold = float(np.percentile(val_scores, config.threshold_pct))

    # --- score: test data through the frozen pipeline ---
    test_series = load_series(test_dir, config)
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
        notes=notes,
    )
