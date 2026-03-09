"""
evaluation_agent.py — Evaluation metrics for AES.

Computes:
  • Quadratic Weighted Kappa (QWK)
  • Root Mean Square Error (RMSE)
  • Pearson Correlation
  • Accuracy (rounded integer scores)
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray,
                              min_rating: Optional[int] = None,
                              max_rating: Optional[int] = None) -> float:
    """
    Compute Quadratic Weighted Kappa.
    Handles both integer and continuous predictions by rounding.
    """
    y_true = np.round(y_true).astype(int)
    y_pred = np.round(y_pred).astype(int)

    if min_rating is None:
        min_rating = min(y_true.min(), y_pred.min())
    if max_rating is None:
        max_rating = max(y_true.max(), y_pred.max())

    num_ratings = max_rating - min_rating + 1
    if num_ratings <= 1:
        return 1.0

    # Clip predictions to [min_rating, max_rating]
    y_true = np.clip(y_true, min_rating, max_rating)
    y_pred = np.clip(y_pred, min_rating, max_rating)

    # Offset to 0-indexed
    y_true -= min_rating
    y_pred -= min_rating

    # Build O matrix (observed)
    O = np.zeros((num_ratings, num_ratings), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t][p] += 1

    # Build weight matrix
    W = np.zeros((num_ratings, num_ratings), dtype=float)
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i][j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    # Build expected matrix
    hist_true = np.sum(O, axis=1)
    hist_pred = np.sum(O, axis=0)
    E = np.outer(hist_true, hist_pred) / len(y_true)

    numerator = np.sum(W * O)
    denominator = np.sum(W * E)

    if denominator == 0:
        return 1.0
    return 1.0 - numerator / denominator


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 0) -> float:
    """Exact match accuracy (with optional ±tolerance in rounded scores)."""
    y_true_r = np.round(y_true).astype(int)
    y_pred_r = np.round(y_pred).astype(int)
    return float(np.mean(np.abs(y_true_r - y_pred_r) <= tolerance))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all evaluation metrics and return as dict."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    qwk = quadratic_weighted_kappa(y_true, y_pred)
    r = rmse(y_true, y_pred)
    pearson = pearson_correlation(y_true, y_pred)
    acc_exact = accuracy(y_true, y_pred, tolerance=0)
    acc_off1 = accuracy(y_true, y_pred, tolerance=1)

    return {
        "qwk": round(qwk, 4),
        "rmse": round(r, 4),
        "pearson": round(pearson, 4),
        "accuracy": round(acc_exact, 4),
        "accuracy_off1": round(acc_off1, 4),
    }


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                   texts: Optional[list] = None) -> dict:
    """Analyse prediction errors by score bucket."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    errors = y_pred - y_true

    analysis = {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "max_overpredict": float(errors.max()),
        "max_underpredict": float(errors.min()),
        "within_1": float(np.mean(np.abs(errors) <= 1)),
        "within_2": float(np.mean(np.abs(errors) <= 2)),
    }

    # Worst-k samples
    worst_idx = np.argsort(np.abs(errors))[::-1][:5]
    worst = []
    for i in worst_idx:
        entry = {
            "idx": int(i),
            "true": float(y_true[i]),
            "pred": float(y_pred[i]),
            "error": float(errors[i]),
        }
        if texts is not None and i < len(texts):
            entry["text_snippet"] = texts[i][:120]
        worst.append(entry)
    analysis["worst_samples"] = worst

    return analysis
