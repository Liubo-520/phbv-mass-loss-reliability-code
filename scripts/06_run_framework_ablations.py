"""Evaluate bounded, monotonic, weighting, precision, and temperature ablations."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data_io import ROOT
from src.evaluation import summarize_prediction_table
from src.models import build_pipeline, sample_weights, xgboost_variant_spec
from src.preprocessing import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


VARIANTS = [
    {
        "name": "direct_unconstrained_row",
        "bounded": False,
        "monotonic": False,
        "weighting": "row",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
    },
    {
        "name": "direct_unconstrained_curve_weighted",
        "bounded": False,
        "monotonic": False,
        "weighting": "curve",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
    },
    {
        "name": "direct_unconstrained_study_weighted",
        "bounded": False,
        "monotonic": False,
        "weighting": "study",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
    },
    {
        "name": "bounded_unconstrained_row",
        "bounded": True,
        "monotonic": False,
        "weighting": "row",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
    },
    {
        "name": "direct_monotonic_row",
        "bounded": False,
        "monotonic": True,
        "weighting": "row",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
    },
    {
        "name": "bounded_monotonic_row",
        "bounded": True,
        "monotonic": True,
        "weighting": "row",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
    },
    {
        "name": "bounded_monotonic_study_weighted",
        "bounded": True,
        "monotonic": True,
        "weighting": "study",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
        "epsilon": 1e-3,
    },
    {
        "name": "bounded_monotonic_study_weighted_epsilon_1e-4",
        "bounded": True,
        "monotonic": True,
        "weighting": "study",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
        "epsilon": 1e-4,
    },
    {
        "name": "bounded_monotonic_study_weighted_epsilon_1e-2",
        "bounded": True,
        "monotonic": True,
        "weighting": "study",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
        "epsilon": 1e-2,
    },
    {
        "name": "bounded_monotonic_study_weighted_cleaned_target",
        "bounded": True,
        "monotonic": True,
        "weighting": "study",
        "target": "weight_loss_cleaned",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
    },
    {
        "name": "bounded_monotonic_study_weighted_cleaned_time",
        "bounded": True,
        "monotonic": True,
        "weighting": "study",
        "target": "weight_loss",
        "time": "cleaned",
        "temperature": True,
        "isotonic": False,
    },
    {
        "name": "bounded_monotonic_study_weighted_no_temperature",
        "bounded": True,
        "monotonic": True,
        "weighting": "study",
        "target": "weight_loss",
        "time": "primary",
        "temperature": False,
        "isotonic": False,
    },
    {
        "name": "direct_unconstrained_row_isotonic",
        "bounded": False,
        "monotonic": False,
        "weighting": "row",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": True,
    },
    {
        "name": "bounded_monotonic_study_weighted_inherited_completion",
        "bounded": True,
        "monotonic": True,
        "weighting": "study",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
        "missingness": "inherited_global",
    },
    {
        "name": "bounded_monotonic_study_weighted_native_missing",
        "bounded": True,
        "monotonic": True,
        "weighting": "study",
        "target": "weight_loss",
        "time": "primary",
        "temperature": True,
        "isotonic": False,
        "missingness": "native_missing",
    },
]


def dense_grid_predictions(pipeline, test: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    outputs = []
    for curve_id, curve in test.groupby("curve_id", sort=False):
        template = curve.iloc[[0]].copy()
        maximum = max(1.0, float(curve["degradation_time_days"].max()))
        grid = np.linspace(0.0, maximum, 101)
        dense = pd.concat([template] * len(grid), ignore_index=True)
        dense["degradation_time_days"] = grid
        pred = pipeline.predict(dense[model_features]).astype(float)
        outputs.append(
            pd.DataFrame(
                {
                    "curve_id": curve_id,
                    "study_id": str(curve["study_id_normalized"].iloc[0]),
                    "degradation_time_days": grid,
                    "y_pred": pred,
                }
            )
        )
    return pd.concat(outputs, ignore_index=True)


def isotonic_observed_predictions(
    dense: pd.DataFrame, observed: pd.DataFrame
) -> np.ndarray:
    output = pd.Series(index=observed.index, dtype=float)
    for curve_id, curve in observed.groupby("curve_id", sort=False):
        grid = dense.loc[dense.curve_id.eq(curve_id)]
        isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
        corrected = isotonic.fit_transform(grid.degradation_time_days, grid.y_pred)
        output.loc[curve.index] = np.interp(
            curve.degradation_time_days.to_numpy(), grid.degradation_time_days, corrected
        )
    return output.loc[observed.index].to_numpy()


def main() -> None:
    table = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv",
        dtype={"study_id_normalized": str, "instance_normalized": str},
    )
    selected = pd.read_csv(
        ROOT / "results" / "tuning" / "selected_hyperparameters.csv",
        dtype={"outer_test_study": str},
    )
    selected = selected.loc[selected.model.eq("xgboost")].set_index("outer_test_study")
    prediction_rows = []
    dense_rows = []
    runtime_rows = []

    for outer_study in sorted(table.study_id_normalized.unique(), key=str):
        base_train = table.loc[table.study_id_normalized.ne(outer_study)].copy()
        base_test = table.loc[table.study_id_normalized.eq(outer_study)].copy()
        params = json.loads(selected.loc[outer_study, "parameters_json"])
        for variant in VARIANTS:
            train = base_train.copy()
            test = base_test.copy()
            if variant["time"] == "cleaned":
                train["degradation_time_days"] = train["degradation_time_cleaned_days"]
                test["degradation_time_days"] = test["degradation_time_cleaned_days"]
            missingness = variant.get("missingness", "restored_fold_local")
            if missingness == "inherited_global":
                for frame in (train, test):
                    frame["t_deg"] = frame["t_deg_source_completed"]
                    for slot in (1, 2, 3):
                        frame[f"additive{slot}_percentage_wt"] = frame[
                            f"additive{slot}_percentage_wt_source_completed"
                        ]
                numeric = [
                    "degradation_time_days",
                    "adjusted_hb_ratio_formulation_mol",
                    "adjusted_hv_ratio_formulation_mol",
                    "t_deg",
                    "additive1_percentage_wt",
                    "additive2_percentage_wt",
                    "additive3_percentage_wt",
                ]
            else:
                numeric = list(NUMERIC_FEATURES)
            if not variant["temperature"]:
                numeric = [
                    column
                    for column in numeric
                    if column not in {"t_deg", "temperature_was_reported"}
                ]
            model_features = numeric + CATEGORICAL_FEATURES
            spec = xgboost_variant_spec(
                params,
                bounded=variant["bounded"],
                monotonic=variant["monotonic"],
                epsilon=variant.get("epsilon", 1e-3),
                name=variant["name"],
            )
            pipeline = build_pipeline(
                spec,
                numeric_features=numeric,
                impute_numeric=missingness != "native_missing",
            )
            weights = sample_weights(train, variant["weighting"])
            fit_start = time.perf_counter()
            pipeline.fit(
                train[model_features],
                train[variant["target"]],
                model__sample_weight=weights,
            )
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            prediction = pipeline.predict(test[model_features])
            dense = dense_grid_predictions(pipeline, test, model_features)
            if variant["isotonic"]:
                prediction = isotonic_observed_predictions(dense, test)
                for curve_id, grid in dense.groupby("curve_id", sort=False):
                    dense.loc[grid.index, "y_pred"] = IsotonicRegression(
                        increasing=True, out_of_bounds="clip"
                    ).fit_transform(grid.degradation_time_days, grid.y_pred)
            predict_seconds = time.perf_counter() - predict_start
            dense.insert(0, "model", variant["name"])
            dense.insert(0, "fold", f"study_{outer_study}")
            dense_rows.append(dense)
            runtime_rows.append(
                {
                    "outer_test_study": outer_study,
                    "model": variant["name"],
                    "fit_seconds": fit_seconds,
                    "predict_seconds_including_dense_grid": predict_seconds,
                    "training_target": variant["target"],
                    "weighting": variant["weighting"],
                    "bounded": variant["bounded"],
                    "monotonic": variant["monotonic"],
                    "isotonic": variant["isotonic"],
                    "primary_raw_decimal_time": variant["time"] == "primary",
                    "temperature_included": variant["temperature"],
                    "missingness_strategy": missingness,
                    "training_targets_clipped_for_logit": int(
                        variant["bounded"]
                        and ((train[variant["target"]] < 0) | (train[variant["target"]] > 100)).sum()
                    ),
                }
            )
            for (_, row), y_pred in zip(test.iterrows(), prediction):
                prediction_rows.append(
                    {
                        "task": "framework_ablation_loso",
                        "fold": f"study_{outer_study}",
                        "model": variant["name"],
                        "row_id": row.row_id,
                        "study_id": outer_study,
                        "curve_id": row.curve_id,
                        "curve_position": int(row.curve_position),
                        "degradation_time_days": float(row.degradation_time_days),
                        "degradation_environment": row.degradation_environment,
                        "y_true": float(row.weight_loss),
                        "y_pred": float(y_pred),
                    }
                )
            print(f"completed ablation {outer_study}/{variant['name']}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    dense_predictions = pd.concat(dense_rows, ignore_index=True)
    metrics, curve_metrics, study_metrics = summarize_prediction_table(predictions)
    prediction_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"
    predictions.to_csv(prediction_dir / "framework_ablation_predictions.csv", index=False)
    dense_predictions.to_csv(prediction_dir / "framework_ablation_dense_grid.csv", index=False)
    metrics.to_csv(metric_dir / "framework_ablation_metrics.csv", index=False)
    curve_metrics.to_csv(metric_dir / "framework_ablation_curve_metrics.csv", index=False)
    study_metrics.to_csv(metric_dir / "framework_ablation_study_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(metric_dir / "framework_ablation_runtime.csv", index=False)


if __name__ == "__main__":
    main()
