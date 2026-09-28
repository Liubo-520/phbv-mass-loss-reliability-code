"""Run expanded-corpus, core-feature, and endpoint-harmonized LOSO analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_io import ROOT
from src.evaluation import summarize_prediction_table
from src.models import build_pipeline, sample_weights, xgboost_variant_spec
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


FIXED_SENSITIVITY_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 3.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
}
CORE_NUMERIC = [
    "degradation_time_days",
    "adjusted_hb_ratio_formulation_mol",
    "adjusted_hv_ratio_formulation_mol",
]
CORE_CATEGORICAL = list(CATEGORICAL_FEATURES)


def run_variant(
    table: pd.DataFrame,
    variant: str,
    numeric: list[str],
    categorical: list[str],
) -> list[dict[str, object]]:
    rows = []
    features = numeric + categorical
    spec = xgboost_variant_spec(
        FIXED_SENSITIVITY_PARAMS,
        bounded=True,
        monotonic=True,
        name=variant,
    )
    for outer_study in sorted(table["study_id_normalized"].unique(), key=str):
        train = table[table["study_id_normalized"].ne(outer_study)].copy()
        test = table[table["study_id_normalized"].eq(outer_study)].copy()
        pipeline = build_pipeline(
            spec,
            numeric_features=numeric,
            categorical_features=categorical,
        )
        pipeline.fit(
            train[features],
            train["weight_loss"],
            model__sample_weight=sample_weights(train, "study"),
        )
        prediction = pipeline.predict(test[features])
        for (_, row), value in zip(test.iterrows(), prediction):
            rows.append(
                {
                    "task": "corpus_endpoint_sensitivity_loso",
                    "fold": f"study_{outer_study}",
                    "model": variant,
                    "row_id": row.row_id,
                    "study_id": str(outer_study),
                    "curve_id": row.curve_id,
                    "curve_position": int(row.curve_position),
                    "degradation_time_days": float(row.degradation_time_days),
                    "degradation_environment": row.degradation_environment,
                    "y_true": float(row.weight_loss),
                    "y_pred": float(value),
                }
            )
        print(f"completed {variant}: held-out study {outer_study}", flush=True)
    return rows


def main() -> None:
    dtype = {"study_id_normalized": str, "instance_normalized": str}
    primary = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv", dtype=dtype
    )
    expanded = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_expanded.csv", dtype=dtype
    )
    endpoint_harmonized = primary[
        ~primary["curve_id"].isin(["study_9__instance_4", "study_9__instance_8"])
    ].copy()

    variants = [
        ("complete_case_primary", primary, NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        ("expanded_full_features", expanded, NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        ("expanded_core_features", expanded, CORE_NUMERIC, CORE_CATEGORICAL),
        ("endpoint_harmonized_excluding_two_study9_conflicts", endpoint_harmonized, NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    ]
    prediction_rows = []
    for name, corpus, numeric, categorical in variants:
        prediction_rows.extend(run_variant(corpus, name, list(numeric), list(categorical)))

    predictions = pd.DataFrame(prediction_rows)
    metrics, curve_metrics, study_metrics = summarize_prediction_table(predictions)
    metrics["study_level_median_mae"] = metrics["model"].map(
        study_metrics.groupby("model")["mae"].median()
    )
    common_studies = set(primary["study_id_normalized"].unique())
    common = study_metrics[study_metrics["study_id"].astype(str).isin(common_studies)]
    common_summary = (
        common.groupby("model", as_index=False)
        .agg(
            common_study_macro_mae=("mae", "mean"),
            common_study_median_mae=("mae", "median"),
            common_studies=("study_id", "nunique"),
        )
    )

    prediction_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"
    predictions.to_csv(prediction_dir / "corpus_endpoint_sensitivity_predictions.csv", index=False)
    metrics.to_csv(metric_dir / "corpus_endpoint_sensitivity_metrics.csv", index=False)
    curve_metrics.to_csv(metric_dir / "corpus_endpoint_sensitivity_curve_metrics.csv", index=False)
    study_metrics.to_csv(metric_dir / "corpus_endpoint_sensitivity_study_metrics.csv", index=False)
    common_summary.to_csv(metric_dir / "corpus_endpoint_common_study_comparison.csv", index=False)
    protocol = {
        "estimator": "bounded_monotonic_fixed_xgboost",
        "parameters": FIXED_SENSITIVITY_PARAMS,
        "weighting": "study_curve_equalized",
        "imputation": "training_fold_median_with_missingness_indicators",
        "core_numeric_features": CORE_NUMERIC,
        "core_categorical_features": CORE_CATEGORICAL,
        "study9_100_minus_y": "not_run_primary_source_does_not_prove_remaining_mass",
    }
    (ROOT / "results" / "corpus_endpoint_sensitivity_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
