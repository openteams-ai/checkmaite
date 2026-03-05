"""kNN-based Bayes Error Rate estimation.

Provides functions for estimating BER upper/lower bounds and building
confusion matrices from k-nearest-neighbor analysis on embeddings.

This code is derived from the DataEval library by ARiA.
Original source: https://github.com/aria-ml/dataeval
Licensed under the MIT License. Copyright (c) 2025 ARiA.

References
----------
[1] Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4)
    https://arxiv.org/abs/1811.06419
[2] Property 2 (Devroye, 1981) for binary case with large k.
"""

import numpy as np
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors

_BER_EPSILON = 1e-12


def _knn_lowerbound(upper: float, num_classes: int, k: int) -> float:
    """Compute BER lower bound from upper bound."""
    if upper <= _BER_EPSILON:
        return 0.0

    if num_classes == 2 and k != 1:
        if k > 5:
            alpha = 0.3399
            beta = 0.9749
            a_k = alpha * np.sqrt(k) / (k - 3.25) * (1 + beta / (np.sqrt(k - 3)))
            return upper / (1 + a_k)
        if k > 2:
            return upper / (1 + (1 / np.sqrt(k)))
        return upper / 2

    m = num_classes
    return ((m - 1) / m) * (1 - np.sqrt(max(0, 1 - (m / (m - 1)) * upper)))


def _fit_knn_predict(embeddings: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Fit kNN and return predicted labels via majority vote.

    Parameters
    ----------
    embeddings
        Array of shape (N, D).
    labels
        Array of shape (N,).
    k
        Number of nearest neighbors (excluding self).

    Returns
    -------
    predicted_labels
        Array of shape (N,) with the majority-vote predicted class for each instance.
    """
    n_samples = len(labels)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_samples), metric="euclidean")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    # Skip self (first neighbor is always the point itself)
    neighbor_labels = labels[indices[:, 1:]]
    return mode(neighbor_labels, axis=1, keepdims=False).mode


def compute_ber_knn(embeddings: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    """Compute kNN BER upper and lower bounds.

    Parameters
    ----------
    embeddings
        Array of shape (N, D).
    labels
        Array of shape (N,).
    k
        Number of nearest neighbors.

    Returns
    -------
    ber_upper, ber_lower
        Upper and lower bounds on Bayes Error Rate.
    """
    n_samples = len(labels)
    num_classes = len(np.unique(labels))

    predicted_labels = _fit_knn_predict(embeddings, labels, k)

    misclassified = np.count_nonzero(predicted_labels != labels)
    ber_upper = float(misclassified / n_samples)
    ber_lower = float(_knn_lowerbound(ber_upper, num_classes, k))

    return ber_upper, ber_lower


def compute_ber_and_confusion(
    embeddings: np.ndarray, labels: np.ndarray, k: int
) -> tuple[float, float, np.ndarray, list[int]]:
    """Compute BER bounds and confusion matrix from a single kNN fit.

    Parameters
    ----------
    embeddings
        Normalized embeddings array of shape (N, D).
    labels
        Label array of shape (N,).
    k
        Number of neighbors to consider.

    Returns
    -------
    ber_upper
        Upper bound on Bayes Error Rate.
    ber_lower
        Lower bound on Bayes Error Rate.
    class_confusion
        Confusion matrix of shape (num_classes, num_classes), rows=true, cols=predicted.
    confusion_labels
        Class IDs corresponding to rows/columns of the confusion matrix.
    """
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    predicted_labels = _fit_knn_predict(embeddings, labels, k)

    misclassified = np.count_nonzero(predicted_labels != labels)
    ber_upper = float(misclassified / n_samples)
    ber_lower = float(_knn_lowerbound(ber_upper, num_classes, k))

    true_idx = np.searchsorted(unique_labels, labels)
    pred_idx = np.searchsorted(unique_labels, predicted_labels)
    class_confusion = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(class_confusion, (true_idx, pred_idx), 1)

    confusion_labels = [int(c) for c in unique_labels]

    return ber_upper, ber_lower, class_confusion, confusion_labels
