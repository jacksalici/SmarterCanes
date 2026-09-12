"""Experiment 2: an autoencoder over aligned step windows, for anomaly detection.

Where experiment 1 answers *how many* steps a recording contains, this one
asks what a step *looks like*. Windows spanning a few steps are cut around
detected strikes, phase-aligned against a common template and z-scored (see
`utils/windows.py`), then a deliberately small MLP autoencoder is trained to
reconstruct them - on normal data only.

The model is therefore only ever shown ordinary gait, and a bottleneck far
narrower than the input forces it to learn the few degrees of freedom that
ordinary gait actually has. Reconstruction error then works as an anomaly
score: a window it cannot reproduce is a window unlike anything it was
trained on. The decision threshold is a high percentile of the error the
model makes on *held-out normal* windows, so it is calibrated on what normal
looks like rather than on any assumption about what anomalies look like -
which matters because the anomaly labels are not settled yet.

Caveat on scale: `data/` currently holds a few hundred windows, far fewer
than the input dimension. The bottleneck, weight decay and the by-session
validation split are all there to keep that honest, and the train/validation
gap in the loss curves is the thing to watch before trusting a score.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from exp.step_count import count_steps
from utils.io import ImuRecording
from utils.windows import NormStats, WindowSet, build_window_set, to_uniform

# Measured over data/: the median interval between detected steps is ~1.94 s,
# so three steps is ~6 s rather than the ~3 s a normal walking cadence would
# suggest - cane-assisted gait here is slow and deliberate.
DEFAULT_WINDOW_S = 6.0
# Alignment search range, ~half a step interval: wide enough to fix a
# mis-anchored window, narrow enough that a window can't align onto the
# neighbouring step and collapse the phase structure we're trying to keep.
DEFAULT_MAX_LAG_S = 1.0
# 25 Hz keeps gait shape and the strike transient while holding the input
# dimension (channels x samples) down, which matters with a few hundred
# windows to learn from.
DEFAULT_TARGET_FS = 25.0


@dataclass
class StepAEConfig:
    # Preprocessing
    window_s: float = DEFAULT_WINDOW_S
    step_hop: int = 1
    target_fs: float = DEFAULT_TARGET_FS
    max_lag_s: float = DEFAULT_MAX_LAG_S
    align_iters: int = 3
    with_dist: bool = False
    # Which windows count as normal training data
    normal_by: str = "event"  # "event" or "ann"
    normal_ann: tuple[int, ...] = (-1,)
    # Model / training
    hidden: tuple[int, ...] = (128, 32)
    latent: int = 8
    epochs: int = 400
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_frac: float = 0.2
    patience: int = 60
    seed: int = 0
    threshold_pct: float = 99.0


@dataclass
class StepAEResult:
    windows: WindowSet
    split: np.ndarray  # (n_windows,) "train" | "val" | "candidate"
    train_loss: np.ndarray  # per epoch
    val_loss: np.ndarray  # per epoch
    best_epoch: int
    scores: np.ndarray  # (n_windows,) mean squared reconstruction error
    channel_error: np.ndarray  # (n_windows, C) per-channel mean squared error
    recon: np.ndarray  # (n_windows, C, L) reconstructions, in normalized units
    threshold: float
    config: StepAEConfig
    trained: bool  # False when scored from a loaded checkpoint
    notes: list[str] = field(default_factory=list)

    @property
    def n_flagged_candidates(self) -> int:
        mask = self.split == "candidate"
        return int(np.sum(self.scores[mask] > self.threshold))

    @property
    def n_candidates(self) -> int:
        return int(np.sum(self.split == "candidate"))

    @property
    def n_flagged_normal(self) -> int:
        mask = self.split != "candidate"
        return int(np.sum(self.scores[mask] > self.threshold))


class StepAutoencoder(nn.Module):
    """A small symmetric MLP autoencoder over a flattened (C, L) window."""

    def __init__(self, n_inputs: int, hidden: tuple[int, ...], latent: int) -> None:
        super().__init__()
        widths = [n_inputs, *hidden]

        encoder: list[nn.Module] = []
        for a, b in zip(widths, widths[1:]):
            encoder += [nn.Linear(a, b), nn.ReLU()]
        encoder.append(nn.Linear(widths[-1], latent))
        self.encoder = nn.Sequential(*encoder)

        widths_back = [latent, *reversed(hidden)]
        decoder: list[nn.Module] = []
        for a, b in zip(widths_back, widths_back[1:]):
            decoder += [nn.Linear(a, b), nn.ReLU()]
        # Linear output: the target is z-scored, so it is not bounded to [0, 1]
        # and squashing it would clip exactly the large excursions that carry
        # the anomaly signal.
        decoder.append(nn.Linear(widths_back[-1], n_inputs))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _normal_mask_fn(config: StepAEConfig):
    """Build the predicate deciding whether a window is normal training data.

    Two label sources, because which one marks an anomaly is still to be
    pinned down: `event` uses the in-recording click marker (a window
    containing one is held out as a candidate), `ann` uses the per-session
    stop annotation in the filename. A recording that carries no label at all
    is treated as normal, so older recordings stay usable.
    """
    if config.normal_by == "event":

        def by_event(series, center: int, window_len: int) -> bool:
            start = center - window_len // 2
            end = start + window_len
            return not np.any((series.event_idx >= start) & (series.event_idx < end))

        return by_event

    if config.normal_by == "ann":
        allowed = set(config.normal_ann)

        def by_ann(series, center: int, window_len: int) -> bool:
            return series.ann is None or series.ann in allowed

        return by_ann

    raise ValueError(f"unknown --normal-by mode: {config.normal_by!r}")


def _split_normal(
    windows: WindowSet, val_frac: float, seed: int
) -> tuple[np.ndarray, list[str]]:
    """Label each window "train" / "val" / "candidate".

    Normal windows are split by *session*, not by window: consecutive windows
    overlap heavily, so a window-level split would put near-duplicates on both
    sides and make the validation loss meaningless. When only one session has
    normal windows there is no such split available, so it falls back to a
    contiguous split in time (still better than random, which would interleave
    overlapping windows) and says so.
    """
    split = np.where(windows.is_normal, "train", "candidate").astype(object)
    notes: list[str] = []

    normal_idx = np.flatnonzero(windows.is_normal)
    if normal_idx.size == 0:
        return split.astype(str), ["no normal windows: every window is a candidate"]

    sessions = np.unique(windows.session_ids[normal_idx])
    n_val_target = max(1, int(round(val_frac * normal_idx.size)))

    if sessions.size >= 2:
        rng = np.random.default_rng(seed)
        order = rng.permutation(sessions)
        val_sessions: list[str] = []
        n_val = 0
        # Leave at least one session for training, however the counts land.
        for session in order[:-1]:
            if n_val >= n_val_target:
                break
            val_sessions.append(session)
            n_val += int(np.sum(windows.session_ids[normal_idx] == session))
        val_mask = np.isin(windows.session_ids, val_sessions) & windows.is_normal
        notes.append(f"validation sessions: {', '.join(sorted(val_sessions))}")
    else:
        order = normal_idx[np.argsort(windows.center_times[normal_idx])]
        val_mask = np.zeros(windows.n_windows, dtype=bool)
        val_mask[order[-n_val_target:]] = True
        notes.append(
            "only one session has normal windows: validation is the last "
            f"{n_val_target} window(s) in time, not a held-out session"
        )

    split[val_mask] = "val"
    return split.astype(str), notes


def _train(
    model: StepAutoencoder,
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    config: StepAEConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit the autoencoder, keeping the weights with the best validation loss."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(config.seed)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    n = x_train.shape[0]
    batch_size = min(config.batch_size, n)

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch_size):
            batch = x_train[order[start : start + batch_size]]
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.shape[0]
        train_losses.append(total / n)

        model.eval()
        with torch.no_grad():
            val_losses.append(
                loss_fn(model(x_val), x_val).item() if x_val.shape[0] else float("nan")
            )

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_state)
    return np.array(train_losses), np.array(val_losses), best_epoch


def _score(model: StepAutoencoder, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window and per-channel reconstruction error, plus the reconstructions."""
    n, n_channels, window_len = x.shape
    flat = torch.from_numpy(x.reshape(n, -1)).float()
    model.eval()
    with torch.no_grad():
        recon_flat = model(flat)
    recon = recon_flat.numpy().reshape(n, n_channels, window_len)
    squared_error = (recon - x) ** 2
    return squared_error.mean(axis=(1, 2)), squared_error.mean(axis=2), recon


def build_windows(
    recs: list[tuple[str, ImuRecording]],
    config: StepAEConfig,
    template: np.ndarray | None = None,
    norm: NormStats | None = None,
) -> WindowSet:
    """Preprocess recordings into aligned, normalized windows."""
    series_list = []
    for session_id, rec in recs:
        step_times = count_steps(rec).step_times
        if step_times.size == 0:
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

    return build_window_set(
        series_list,
        window_s=config.window_s,
        step_hop=config.step_hop,
        max_lag_s=config.max_lag_s,
        align_iters=config.align_iters,
        normal_mask_fn=_normal_mask_fn(config),
        template=template,
        norm=norm,
    )


def run_step_ae(
    recs: list[tuple[str, ImuRecording]],
    config: StepAEConfig,
    load_path: Path | None = None,
    save_path: Path | None = None,
) -> StepAEResult:
    """Preprocess, train (or load) and score.

    With `load_path`, the saved template and normalization statistics are
    reused for preprocessing as well as the weights, so a checkpoint scores
    new recordings through exactly the pipeline it was trained on.
    """
    torch.manual_seed(config.seed)

    checkpoint = None
    template = norm = None
    if load_path is not None:
        checkpoint = torch.load(load_path, weights_only=False)
        config = checkpoint["config"]
        template = checkpoint["template"]
        norm = NormStats(mean=checkpoint["norm_mean"], std=checkpoint["norm_std"])

    windows = build_windows(recs, config, template=template, norm=norm)
    split, notes = _split_normal(windows, config.val_frac, config.seed)

    n_inputs = windows.x.shape[1] * windows.x.shape[2]
    model = StepAutoencoder(n_inputs, tuple(config.hidden), config.latent)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])
        train_loss = checkpoint["train_loss"]
        val_loss = checkpoint["val_loss"]
        best_epoch = checkpoint["best_epoch"]
    else:
        x_train = torch.from_numpy(
            windows.x[split == "train"].reshape((split == "train").sum(), -1)
        ).float()
        x_val = torch.from_numpy(
            windows.x[split == "val"].reshape((split == "val").sum(), -1)
        ).float()
        if x_train.shape[0] == 0:
            raise ValueError("no training windows: every window was held out as a candidate")
        train_loss, val_loss, best_epoch = _train(model, x_train, x_val, config)

    scores, channel_error, recon = _score(model, windows.x)

    # Calibrate on held-out normal windows if there are any - the error on
    # windows the model was fitted to is optimistically low.
    calib_mask = split == "val"
    if not calib_mask.any():
        calib_mask = split == "train"
        notes.append("no validation windows: threshold calibrated on training windows")
    threshold = float(np.percentile(scores[calib_mask], config.threshold_pct))

    if save_path is not None and checkpoint is None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": config,
                "channels": windows.channels,
                "window_len": windows.window_len,
                "template": windows.template,
                "norm_mean": windows.norm.mean,
                "norm_std": windows.norm.std,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_epoch": best_epoch,
                "threshold": threshold,
            },
            save_path,
        )

    return StepAEResult(
        windows=windows,
        split=split,
        train_loss=train_loss,
        val_loss=val_loss,
        best_epoch=best_epoch,
        scores=scores,
        channel_error=channel_error,
        recon=recon,
        threshold=threshold,
        config=config,
        trained=checkpoint is None,
        notes=notes,
    )


def config_summary(config: StepAEConfig) -> str:
    return ", ".join(f"{k}={v}" for k, v in asdict(config).items())
