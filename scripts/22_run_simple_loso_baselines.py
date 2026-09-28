"""Evaluate transparent baselines on exactly the primary LOSO folds."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

from src.data_io import ROOT
from src.evaluation import summarize_dense_physical_predictions, summarize_prediction_table
from src.models import ModelSpec, build_pipeline, sample_weights
from src.preprocessing import MODEL_FEATURES


BASELINE_TASK = "simple_baseline_loso"


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered, ordered_weights = values[order], weights[order]
    return float(ordered[np.searchsorted(np.cumsum(ordered_weights), ordered_weights.sum() / 2)])


def fit_isotonic(train: pd.DataFrame, weights: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        train["degradation_time_days"].to_numpy(float),
        train["weight_loss"].to_numpy(float),
        sample_weight=weights,
    )


def environment_isotonic_models(
    train: pd.DataFrame, weights: np.ndarray
) -> tuple[IsotonicRegression, dict[str, IsotonicRegression]]:
    global_model = fit_isotonic(train, weights)
    models = {}
    for environment, index in train.groupby("degradation_environment", sort=False).groups.items():
        subset = train.loc[index]
        if subset["degradation_time_days"].nunique() >= 2:
            local_weights = pd.Series(weights, index=train.index).loc[index].to_numpy()
            models[str(environment)] = fit_isotonic(subset, local_weights)
    return global_model, models


def predict_environment_isotonic(
    frame: pd.DataFrame,
    global_model: IsotonicRegression,
    models: dict[str, IsotonicRegression],
) -> np.ndarray:
    output = np.empty(len(frame), dtype=float)
    for environment, index in frame.groupby("degradation_environment", sort=False).groups.items():
        model = models.get(str(environment), global_model)
        positions = frame.index.get_indexer(index)
        output[positions] = model.predict(frame.loc[index, "degradation_time_days"])
    return output


def dense_test_grid(test: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for curve_id, curve in test.groupby("curve_id", sort=False):
        grid = np.linspace(0.0, max(1.0, float(curve["degradation_time_days"].max())), 101)
        dense = pd.concat([curve.iloc[[0]]] * len(grid), ignore_index=True)
        dense["degradation_time_days"] = grid
        dense["curve_id"] = curve_id
        outputs.append(dense)
    return pd.concat(outputs, ignore_index=True)


def append_predictions(
    rows: list[dict[str, object]], test: pd.DataFrame, prediction: np.ndarray, model: str, fold: str
) -> None:
    for (_, row), value in zip(test.iterrows(), prediction):
        rows.append(
            {
                "task": BASELINE_TASK,
                "fold": fold,
                "model": model,
                "row_id": row.row_id,
                "study_id": str(row.study_id_normalized),
                "curve_id": row.curve_id,
                "curve_position": int(row.curve_position),
                "degradation_time_days": float(row.degradation_time_days),
                "degradation_environment": row.degradation_environment,
                "y_true": float(row.weight_loss),
                "y_pred": float(value),
            }
        )


def main() -> None:
    table = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv",
        dtype={"study_id_normalized": str, "instance_normalized": str},
    )
    prediction_rows: list[dict[str, object]] = []
    dense_rows = []

    ridge_spec = ModelSpec("ridge", lambda: Ridge(alpha=10.0), scale_numeric=True)
    for outer_study in sorted(table["study_id_normalized"].unique(), key=str):
        train = table[table["study_id_normalized"].ne(outer_study)].copy()
        test = table[table["study_id_normalized"].eq(outer_study)].copy()
        weights = sample_weights(train, "study")
        mean_value = float(np.average(train["weight_loss"], weights=weights))
        median_value = weighted_median(train["weight_loss"].to_numpy(float), weights)
        time_model = fit_isotonic(train, weights)
        env_global, env_models = environment_isotonic_models(train, weights)
        ridge = build_pipeline(ridge_spec)
        ridge.fit(train[MODEL_FEATURES], train["weight_loss"], model__sample_weight=weights)

        observed_predictions = {
            "global_mean": np.full(len(test), mean_value),
            "global_median": np.full(len(test), median_value),
            "time_only_isotonic": time_model.predict(test["degradation_time_days"]),
            "environment_time_isotonic": predict_environment_isotonic(test, env_global, env_models),
            "ridge_all_features": ridge.predict(test[MODEL_FEATURES]),
        }
        fold = f"study_{outer_study}"
        for model, prediction in observed_predictions.items():
            append_predictions(prediction_rows, test, np.asarray(prediction), model, fold)

        dense = dense_test_grid(test)
        dense_predictions = {
            "global_mean": np.full(len(dense), mean_value),
            "global_median": np.full(len(dense), median_value),
            "time_only_isotonic": time_model.predict(dense["degradation_time_days"]),
            "environment_time_isotonic": predict_environment_isotonic(dense, env_global, env_models),
            "ridge_all_features": ridge.predict(dense[MODEL_FEATURES]),
        }
        for model, prediction in dense_predictions.items():
            dense_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "model": model,
                        "curve_id": dense["curve_id"].to_numpy(),
                        "study_id": outer_study,
                        "degradation_time_days": dense["degradation_time_days"].to_numpy(),
                        "y_pred": prediction,
                    }
                )
            )
        print(f"completed simple baselines for held-out study {outer_study}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    dense = pd.concat(dense_rows, ignore_index=True)

    primary_path = ROOT / "results" / "predictions" / "framework_ablation_predictions.csv"
    primary_dense_path = ROOT / "results" / "predictions" / "framework_ablation_dense_grid.csv"
    if primary_path.exists() and primary_dense_path.exists():
        primary = pd.read_csv(primary_path, dtype={"study_id": str})
        primary = primary[primary["model"].eq("bounded_monotonic_study_weighted")].copy()
        expected_time = table.set_index("row_id")["degradation_time_days"]
        if not np.allclose(primary["degradation_time_days"], primary["row_id"].map(expected_time)):
            raise RuntimeError("Framework predictions are stale relative to primary raw-decimal time.")
        primary["task"] = BASELINE_TASK
        primary["model"] = "bounded_monotonic_xgboost_primary"
        predictions = pd.concat([predictions, primary[predictions.columns]], ignore_index=True)
        primary_dense = pd.read_csv(primary_dense_path, dtype={"study_id": str})
        primary_dense = primary_dense[
            primary_dense["model"].eq("bounded_monotonic_study_weighted")
        ].copy()
        primary_dense["model"] = "bounded_monotonic_xgboost_primary"
        dense = pd.concat([dense, primary_dense[dense.columns]], ignore_index=True)

    metrics, curve_metrics, study_metrics = summarize_prediction_table(predictions)
    study_medians = study_metrics.groupby("model")["mae"].median()
    metrics["study_level_median_mae"] = metrics["model"].map(study_medians)
    baseline = metrics[metrics["model"].eq("global_mean")].iloc[0]
    for metric in ("point_mae", "macro_curve_mae", "macro_study_mae", "study_level_median_mae"):
        metrics[f"{metric}_skill_vs_global_mean"] = 1.0 - metrics[metric] / baseline[metric]
    dense_physical = summarize_dense_physical_predictions(dense)
    metrics = metrics.merge(
        dense_physical.add_prefix("dense_").rename(columns={"dense_model": "model"}),
        on="model",
        how="left",
    )

    prediction_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"
    predictions.to_csv(prediction_dir / "simple_loso_baseline_predictions.csv", index=False)
    dense.to_csv(prediction_dir / "simple_loso_baseline_dense_grid.csv", index=False)
    metrics.to_csv(metric_dir / "simple_loso_baselines.csv", index=False)
    curve_metrics.to_csv(metric_dir / "simple_loso_baseline_curve_metrics.csv", index=False)
    study_metrics.to_csv(metric_dir / "simple_loso_baseline_study_metrics.csv", index=False)
    dense_physical.to_csv(metric_dir / "simple_loso_baseline_physical_metrics.csv", index=False)


if __name__ == "__main__":
    main()
