"""Checks for the reviewer-requested split, weighting, and shift analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_io import ROOT
from src.evaluation import point_metrics
from src.models import sample_weights


def test_study_weighting_strategies_and_cap() -> None:
    table = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv",
        dtype={"study_id_normalized": str},
    )
    train = table.loc[table.study_id_normalized.ne("9")].copy()
    for strategy in ["study_only", "study_curve"]:
        weighted = train[["study_id_normalized"]].copy()
        weighted["weight"] = sample_weights(train, strategy)
        totals = weighted.groupby("study_id_normalized").weight.sum().to_numpy()
        assert np.allclose(totals, totals[0])
        assert np.isclose(weighted.weight.mean(), 1.0)

    uncapped = sample_weights(train, "study_curve")
    capped = sample_weights(train, "study_curve_capped95")
    assert np.isclose(capped.mean(), 1.0)
    assert capped.max() < uncapped.max()


def test_repeated_split_metrics_come_from_saved_predictions() -> None:
    predictions = pd.read_csv(
        ROOT / "results" / "predictions" / "repeated_split_predictions.csv"
    )
    metrics = pd.read_csv(ROOT / "results" / "metrics" / "repeated_split_metrics.csv")
    assert predictions.groupby("task").repeat.nunique().to_dict() == {
        "repeated_curve_grouped": 50,
        "repeated_random_row": 50,
    }
    first = predictions.loc[
        predictions.task.eq("repeated_random_row") & predictions.repeat.eq(1)
    ]
    expected = point_metrics(first.y_true.to_numpy(), first.y_pred.to_numpy())
    observed = metrics.loc[
        metrics.task.eq("repeated_random_row") & metrics.repeat.eq(1)
    ].iloc[0]
    assert np.isclose(observed.mae, expected["mae"])
    assert np.isclose(observed.rmse, expected["rmse"])
    assert np.isclose(observed.r2, expected["r2"])


def test_focal_shift_diagnostics_capture_distinct_failures() -> None:
    shift = pd.read_csv(ROOT / "results" / "metrics" / "study_level_shift_summary.csv")
    shift = shift.set_index("study_id")
    assert shift.loc[9, "numeric_features_outside_training_range_count"] == 0
    assert shift.loc[9, "observed_linear_slope_mean"] < 0
    assert shift.loc[19, "numeric_features_outside_training_range_count"] >= 1
    assert shift.loc[19, "unseen_category_feature_count"] >= 1
    assert shift.loc[12, "mae"] < shift.loc[19, "mae"]
    assert shift.loc[14, "r2_for_reporting"] != shift.loc[14, "r2_for_reporting"]
