"""Study-level uncertainty and paired model comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def paired_bootstrap_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap paired study-level mean MAE differences (A minus B)."""
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) != len(values_b):
        raise ValueError("Paired arrays must have equal length")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values_a), size=(iterations, len(values_a)))
    differences = (values_a[indices] - values_b[indices]).mean(axis=1)
    return {
        "mean_difference_a_minus_b": float(np.mean(values_a - values_b)),
        "median_difference_a_minus_b": float(np.median(values_a - values_b)),
        "bootstrap_mean_difference_ci_lower": float(np.quantile(differences, 0.025)),
        "bootstrap_mean_difference_ci_upper": float(np.quantile(differences, 0.975)),
    }


def rank_biserial_effect(differences: np.ndarray) -> float:
    """Calculate matched-pairs rank-biserial correlation, excluding exact zeros."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[differences != 0]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / ranks.sum())


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Apply Holm's step-down family-wise error correction."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (len(p_values) - np.arange(len(p_values))) * p_values[order])
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def compare_models_by_study(
    study_metrics: pd.DataFrame, comparisons: list[tuple[str, str]]
) -> pd.DataFrame:
    """Run paired Wilcoxon and bootstrap comparisons with study as the unit."""
    pivot = study_metrics.pivot(index="study_id", columns="model", values="mae")
    rows = []
    for index, (model_a, model_b) in enumerate(comparisons):
        pair = pivot[[model_a, model_b]].dropna()
        differences = pair[model_a].to_numpy() - pair[model_b].to_numpy()
        if np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "positive_difference_means_b_lower_mae": True,
                "n_studies": len(pair),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
                "rank_biserial_a_minus_b": rank_biserial_effect(differences),
                **paired_bootstrap_difference(
                    pair[model_a].to_numpy(), pair[model_b].to_numpy(), seed=42 + index
                ),
            }
        )
    output = pd.DataFrame(rows)
    output["holm_adjusted_p_value"] = holm_adjust(output.wilcoxon_p_value.to_numpy())
    return output


def bootstrap_model_mean_ci(
    study_metrics: pd.DataFrame, iterations: int = 10_000, seed: int = 42
) -> pd.DataFrame:
    """Calculate model-wise study bootstrap CIs for macro-study MAE."""
    rows = []
    rng = np.random.default_rng(seed)
    for model, group in study_metrics.groupby("model"):
        values = group.mae.to_numpy(dtype=float)
        indices = rng.integers(0, len(values), size=(iterations, len(values)))
        means = values[indices].mean(axis=1)
        rows.append(
            {
                "model": model,
                "n_studies": len(values),
                "macro_study_mae": float(values.mean()),
                "bootstrap_ci_lower": float(np.quantile(means, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(means, 0.975)),
            }
        )
    return pd.DataFrame(rows)

