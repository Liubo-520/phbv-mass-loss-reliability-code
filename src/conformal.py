"""Group-aware split-conformal residual intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_conformal_quantile(
    residuals: np.ndarray, weights: np.ndarray, coverage: float
) -> float:
    """Return a conservative finite-weight conformal residual quantile."""
    residuals = np.asarray(residuals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between 0 and 1")
    if len(residuals) == 0 or len(residuals) != len(weights):
        raise ValueError("residual and weight arrays must be non-empty and equally sized")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive total mass")
    order = np.argsort(residuals, kind="mergesort")
    sorted_residuals = residuals[order]
    normalized = weights[order] / weights.sum()
    adjusted_level = min(1.0, coverage + float(normalized.max()))
    index = int(np.searchsorted(np.cumsum(normalized), adjusted_level, side="left"))
    return float(sorted_residuals[min(index, len(sorted_residuals) - 1)])


def interval_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, coverage: float
) -> np.ndarray:
    """Calculate the central prediction-interval score for each observation."""
    alpha = 1.0 - coverage
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return (
        upper
        - lower
        + (2.0 / alpha) * (lower - y_true) * (y_true < lower)
        + (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    )


def summarize_intervals(predictions: pd.DataFrame, coverages: list[float]) -> pd.DataFrame:
    """Summarize empirical coverage, width, and interval score."""
    rows = []
    for coverage in coverages:
        suffix = str(int(round(coverage * 100)))
        lower = predictions[f"lower_{suffix}"].to_numpy()
        upper = predictions[f"upper_{suffix}"].to_numpy()
        y_true = predictions["y_true"].to_numpy()
        covered = (y_true >= lower) & (y_true <= upper)
        width = upper - lower
        rows.append(
            {
                "nominal_coverage": coverage,
                "empirical_coverage": float(covered.mean()),
                "mean_interval_width": float(width.mean()),
                "median_interval_width": float(np.median(width)),
                "mean_interval_score": float(interval_score(y_true, lower, upper, coverage).mean()),
                "n": len(predictions),
            }
        )
    return pd.DataFrame(rows)

