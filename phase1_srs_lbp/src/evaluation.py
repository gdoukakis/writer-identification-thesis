from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalMetrics:
    top1: float
    mAP: float


def pairwise_squared_euclidean(x: np.ndarray) -> np.ndarray:
    """Return the within-set squared Euclidean distance matrix."""
    x = np.asarray(x, dtype=np.float64)
    squared_norms = np.sum(x * x, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * (x @ x.T)
    return np.maximum(distances, 0.0)


def average_precision(relevances: np.ndarray) -> float:
    """Calculate AP from the relevance values of the final gallery ranking."""
    relevances = np.asarray(relevances, dtype=np.int64)
    number_relevant = int(relevances.sum())
    if number_relevant == 0:
        return 0.0

    cumulative_relevant = np.cumsum(relevances)
    precision_at_rank = cumulative_relevant / np.arange(
        1, relevances.size + 1
    )
    return float(
        np.sum(precision_at_rank * relevances) / number_relevant
    )


def evaluate_retrieval_within_set(
    descriptors: np.ndarray,
    writer_ids: np.ndarray,
) -> RetrievalMetrics:
    """
    Evaluate leave-one-image-out retrieval.

    The query index is removed completely from its gallery before Top-1 and AP
    are calculated.
    """
    descriptors = np.asarray(descriptors, dtype=np.float32)
    writer_ids = np.asarray(writer_ids)

    if descriptors.ndim != 2:
        raise ValueError(
            "descriptors must have shape (n_samples, n_features)."
        )
    if writer_ids.ndim != 1:
        raise ValueError("writer_ids must be one-dimensional.")
    if descriptors.shape[0] != writer_ids.shape[0]:
        raise ValueError(
            "descriptors and writer_ids must contain the same samples."
        )

    distances = pairwise_squared_euclidean(descriptors)
    all_indices = np.arange(descriptors.shape[0])

    top1_correct = np.empty(descriptors.shape[0], dtype=np.float64)
    average_precisions = np.empty(
        descriptors.shape[0],
        dtype=np.float64,
    )

    for query_index in range(descriptors.shape[0]):
        gallery_mask = all_indices != query_index
        gallery_indices = all_indices[gallery_mask]

        order = np.argsort(
            distances[query_index, gallery_mask],
            kind="mergesort",
        )
        ranked_indices = gallery_indices[order]
        relevant = (
            writer_ids[ranked_indices] == writer_ids[query_index]
        )

        top1_correct[query_index] = float(relevant[0])
        average_precisions[query_index] = average_precision(relevant)

    return RetrievalMetrics(
        top1=float(top1_correct.mean()),
        mAP=float(average_precisions.mean()),
    )
