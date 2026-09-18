"""Binary-classification metrics for anomaly scoring, over numpy.

Scores are anomaly scores - higher means more anomalous - so a positive
window is one whose score exceeds the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdMetrics:
    """Counts and rates at one decision threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else float("nan")

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.recall + self.specificity)


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> ThresholdMetrics:
    """Confusion counts for `scores > threshold` predicting `y_true == 1`."""
    pred = scores > threshold
    actual = y_true.astype(bool)
    return ThresholdMetrics(
        threshold=float(threshold),
        tp=int(np.sum(pred & actual)),
        fp=int(np.sum(pred & ~actual)),
        tn=int(np.sum(~pred & ~actual)),
        fn=int(np.sum(~pred & actual)),
    )


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False-positive and true-positive rates over every distinct threshold.

    Steps through groups of equal scores at once, so ties aren't credited
    with an ordering they don't have.
    """
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order].astype(bool)
    s = scores[order]

    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    distinct = np.flatnonzero(np.diff(s)) if s.size > 1 else np.array([], dtype=int)
    ends = np.r_[distinct, s.size - 1]

    tps = np.cumsum(y)[ends]
    fps = np.cumsum(~y)[ends]
    return np.r_[0.0, fps / n_neg], np.r_[0.0, tps / n_pos]


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, by trapezoid over the tie-aware curve."""
    fpr, tpr = roc_curve(y_true, scores)
    if fpr.size < 2:
        return float("nan")
    return float(np.trapezoid(tpr, fpr))


def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recall and precision over every distinct threshold, recall ascending."""
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order].astype(bool)
    s = scores[order]

    n_pos = int(y.sum())
    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    distinct = np.flatnonzero(np.diff(s)) if s.size > 1 else np.array([], dtype=int)
    ends = np.r_[distinct, s.size - 1]

    tps = np.cumsum(y)[ends]
    fps = np.cumsum(~y)[ends]
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / n_pos
    return recall, precision


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUPRC as the step-wise average precision (not trapezoidal - that would
    interpolate between operating points that don't exist).
    """
    recall, precision = pr_curve(y_true, scores)
    if int(np.sum(y_true)) == 0:
        return float("nan")
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_f1(y_true: np.ndarray, scores: np.ndarray) -> ThresholdMetrics:
    """The threshold maximizing F1 - an oracle upper bound, since it peeks
    at the test labels.
    """
    candidates = np.unique(scores)
    cuts = np.r_[candidates[0] - 1e-12, (candidates[:-1] + candidates[1:]) / 2, candidates[-1]]
    best = max((threshold_metrics(y_true, scores, c) for c in cuts), key=lambda m: m.f1)
    return best


def prevalence(y_true: np.ndarray) -> float:
    """Positive rate - the AUPRC a random-scoring detector would reach."""
    return float(np.mean(y_true)) if y_true.size else float("nan")
