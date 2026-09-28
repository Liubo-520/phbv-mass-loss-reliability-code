"""Calculate trajectory metrics and study-level statistical comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_io import ROOT
from src.evaluation import curve_trajectory_metrics
from src.statistics import bootstrap_model_mean_ci, compare_models_by_study


COMPARISONS = [
    ("direct_unconstrained_row", "bounded_unconstrained_row"),
    ("bounded_unconstrained_row", "bounded_monotonic_row"),
    ("bounded_monotonic_row", "bounded_monotonic_study_weighted"),
    (
        "bounded_monotonic_study_weighted",
        "bounded_monotonic_study_weighted_cleaned_target",
    ),
    ("bounded_monotonic_study_weighted", "bounded_monotonic_study_weighted_cleaned_time"),
    (
        "bounded_monotonic_study_weighted",
        "bounded_monotonic_study_weighted_no_temperature",
    ),
    ("direct_unconstrained_row", "direct_unconstrained_row_isotonic"),
]


def main() -> None:
    predictions = pd.read_csv(
        ROOT / "results" / "predictions" / "framework_ablation_predictions.csv",
        dtype={"study_id": str},
    )
    study_metrics = pd.read_csv(
        ROOT / "results" / "metrics" / "framework_ablation_study_metrics.csv",
        dtype={"study_id": str},
    )
    trajectories = curve_trajectory_metrics(predictions)
    comparisons = compare_models_by_study(study_metrics, COMPARISONS)
    confidence_intervals = bootstrap_model_mean_ci(study_metrics)

    metrics_dir = ROOT / "results" / "metrics"
    statistics_dir = ROOT / "results" / "statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(metrics_dir / "framework_trajectory_metrics.csv", index=False)
    comparisons.to_csv(statistics_dir / "model_comparisons.csv", index=False)
    confidence_intervals.to_csv(statistics_dir / "model_macro_study_bootstrap_ci.csv", index=False)


if __name__ == "__main__":
    main()
