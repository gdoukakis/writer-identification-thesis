from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


def pairwise_squared_euclidean(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute squared Euclidean distance matrix.

    If Y is None, computes distances within X (N x N).
    Otherwise computes distances between X (N x D) and Y (M x D): (N x M).
    """
    if Y is None:
        Y = X
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)

    XX = np.sum(X * X, axis=1, keepdims=True)          # (N, 1)
    YY = np.sum(Y * Y, axis=1, keepdims=True).T        # (1, M)
    D = XX + YY - 2.0 * (X @ Y.T)                      # (N, M)
    return np.maximum(D, 0.0)


def average_precision(sorted_relevances: np.ndarray) -> float:
    """
    Compute AP given a binary relevance array ordered by ranking (1=relevant, 0=not).
    """
    rel = sorted_relevances.astype(np.int32, copy=False)
    n_rel = int(rel.sum())
    if n_rel == 0:
        return 0.0

    cumsum = np.cumsum(rel)
    precision_at_k = cumsum / (np.arange(rel.size) + 1.0)
    ap = float((precision_at_k * rel).sum() / n_rel)
    return ap


@dataclass
class RetrievalMetrics:
    top1: float
    mAP: float


def top1_accuracy_cross_set(D_qg: np.ndarray, yQ: np.ndarray, yG: np.ndarray) -> float:
    """
    Top-1 retrieval accuracy for cross-set retrieval (queries vs gallery).

    For each query i, pick nearest gallery sample and compare writer labels.
    """
    nn = np.argmin(D_qg, axis=1)
    correct = (yG[nn] == yQ).astype(np.float32)
    return float(correct.mean())


def mean_average_precision_cross_set(D_qg: np.ndarray, yQ: np.ndarray, yG: np.ndarray) -> float:
    """
    mAP for cross-set retrieval (queries vs gallery).

    For each query i:
      - rank all gallery samples by ascending distance
      - relevant = same writer label in the gallery
      - compute AP over ranked list
    """
    aps = []
    for i in range(D_qg.shape[0]):
        order = np.argsort(D_qg[i], kind="mergesort")
        rel = (yG[order] == yQ[i]).astype(np.int32)
        aps.append(average_precision(rel))
    return float(np.mean(aps))


# Cross-set retrieval evaluation (NOT used in Phase 1 ICDAR2017 run,
# but kept for completeness and future experiments).
def evaluate_retrieval(
    Q: np.ndarray,
    yQ: np.ndarray,
    G: np.ndarray,
    yG: np.ndarray,
) -> RetrievalMetrics:
    """
    Evaluate retrieval metrics for cross-set protocol: queries(Q) against gallery(G).
    """
    yQ = np.asarray(yQ)
    yG = np.asarray(yG)

    D = pairwise_squared_euclidean(Q, G)  # (nQ, nG)
    top1 = top1_accuracy_cross_set(D, yQ, yG)
    mAP = mean_average_precision_cross_set(D, yQ, yG)
    return RetrievalMetrics(top1=top1, mAP=mAP)


# Optional: keep within-set evaluation for sanity checks / debugging
def evaluate_retrieval_within_set(X: np.ndarray, y: np.ndarray) -> RetrievalMetrics:
    """
    Evaluate retrieval within the same set (query=gallery) excluding self-match.
    Useful only for debugging; NOT the ICDAR2017 protocol we need.
    """
    y = np.asarray(y)
    D = pairwise_squared_euclidean(X)

    # Exclude self
    D2 = D.copy()
    np.fill_diagonal(D2, np.inf)

    nn = np.argmin(D2, axis=1)
    top1 = float((y[nn] == y).astype(np.float32).mean())

    aps = []
    for i in range(D2.shape[0]):
        order = np.argsort(D2[i], kind="mergesort")
        rel = (y[order] == y[i]).astype(np.int32)
        aps.append(average_precision(rel))
    mAP = float(np.mean(aps))

    return RetrievalMetrics(top1=top1, mAP=mAP)
