import numpy as np

from src.statistics import holm_adjust, paired_bootstrap_difference, rank_biserial_effect


def test_holm_adjustment_is_bounded_and_not_below_raw_p_values():
    raw = np.array([0.01, 0.04, 0.20])
    adjusted = holm_adjust(raw)
    assert ((adjusted >= raw) & (adjusted <= 1)).all()


def test_paired_bootstrap_and_rank_biserial_direction():
    a = np.array([5.0, 6.0, 7.0, 8.0])
    b = np.array([4.0, 5.0, 6.0, 7.0])
    result = paired_bootstrap_difference(a, b, iterations=1000, seed=42)
    assert result["mean_difference_a_minus_b"] > 0
    assert rank_biserial_effect(a - b) == 1.0
