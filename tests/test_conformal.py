import numpy as np

from src.conformal import interval_score, weighted_conformal_quantile


def test_weighted_conformal_quantile_is_observed_and_monotone_in_coverage():
    residuals = np.array([1.0, 2.0, 4.0, 8.0])
    weights = np.ones(4)
    q80 = weighted_conformal_quantile(residuals, weights, 0.80)
    q95 = weighted_conformal_quantile(residuals, weights, 0.95)
    assert q80 in residuals
    assert q95 in residuals
    assert q95 >= q80


def test_interval_score_penalizes_missed_observations():
    inside = interval_score(np.array([5.0]), np.array([0.0]), np.array([10.0]), 0.9)
    outside = interval_score(np.array([20.0]), np.array([0.0]), np.array([10.0]), 0.9)
    assert outside[0] > inside[0]
