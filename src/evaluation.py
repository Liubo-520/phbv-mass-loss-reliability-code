"""Metrics calculated from a single saved held-out prediction table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics with explicit RMSE."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
    }


def _group_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_id, group in predictions.groupby(group_column, sort=False):
        metrics = point_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        rows.append({group_column: group_id, "n": len(group), **metrics})
    return pd.DataFrame(rows)


def physical_metrics(predictions: pd.DataFrame, tolerance: float = 1e-9) -> dict[str, float]:
    """Calculate observed-grid bounds and monotonicity diagnostics."""
    pred = predictions["y_pred"].to_numpy()
    decreases: list[float] = []
    comparisons = 0
    curves_with_violation = 0
    for _, curve in predictions.groupby("curve_id"):
        ordered = curve.sort_values(["degradation_time_days", "curve_position"])
        delta = np.diff(ordered["y_pred"].to_numpy())
        negative = -delta[delta < -tolerance]
        comparisons += len(delta)
        if len(negative):
            curves_with_violation += 1
            decreases.extend(negative.tolist())
    return {
        "negative_prediction_fraction": float(np.mean(pred < 0)),
        "above_100_prediction_fraction": float(np.mean(pred > 100)),
        "monotonicity_violation_fraction": float(len(decreases) / comparisons) if comparisons else 0.0,
        "curves_with_violation_fraction": float(
            curves_with_violation / predictions["curve_id"].nunique()
        ),
        "mean_violation_magnitude": float(np.mean(decreases)) if decreases else 0.0,
        "maximum_violation_magnitude": float(np.max(decreases)) if decreases else 0.0,
    }


def summarize_prediction_table(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize each task/model using identical held-out predictions."""
    summaries = []
    curve_outputs = []
    study_outputs = []
    for (task, model), group in predictions.groupby(["task", "model"], sort=True):
        point = point_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        curves = _group_metrics(group, "curve_id")
        studies = _group_metrics(group, "study_id")
        curves.insert(0, "model", model)
        curves.insert(0, "task", task)
        studies.insert(0, "model", model)
        studies.insert(0, "task", task)
        curve_outputs.append(curves)
        study_outputs.append(studies)
        summaries.append(
            {
                "task": task,
                "model": model,
                "n_predictions": len(group),
                **{f"point_{key}": value for key, value in point.items()},
                "macro_curve_mae": float(curves["mae"].mean()),
                "macro_curve_rmse": float(curves["rmse"].mean()),
                "macro_study_mae": float(studies["mae"].mean()),
                "macro_study_rmse": float(studies["rmse"].mean()),
                **physical_metrics(group),
            }
        )
    return (
        pd.DataFrame(summaries),
        pd.concat(curve_outputs, ignore_index=True),
        pd.concat(study_outputs, ignore_index=True),
    )


def summarize_dense_physical_predictions(dense: pd.DataFrame) -> pd.DataFrame:
    """Summarize physical validity on dense, fixed-descriptor time grids."""
    rows = []
    for model, group in dense.groupby("model", sort=True):
        violations = []
        maximum_steps = []
        curves_with_violation = 0
        comparisons = 0
        time_zero = []
        for _, curve in group.groupby(["fold", "curve_id"], sort=False):
            ordered = curve.sort_values("degradation_time_days")
            prediction = ordered["y_pred"].to_numpy(dtype=float)
            difference = np.diff(prediction)
            negative = -difference[difference < -1e-9]
            comparisons += len(difference)
            if len(negative):
                curves_with_violation += 1
                violations.extend(negative.tolist())
            if len(difference):
                maximum_steps.append(float(np.max(np.abs(difference))))
            time_zero.append(float(prediction[0]))
        rows.append(
            {
                "model": model,
                "dense_grid_predictions": len(group),
                "curves": int(group.groupby(["fold", "curve_id"]).ngroups),
                "negative_prediction_fraction": float((group.y_pred < 0).mean()),
                "above_100_prediction_fraction": float((group.y_pred > 100).mean()),
                "monotonicity_violation_fraction": float(len(violations) / comparisons),
                "curves_with_violation_fraction": float(
                    curves_with_violation / group.groupby(["fold", "curve_id"]).ngroups
                ),
                "mean_violation_magnitude": float(np.mean(violations)) if violations else 0.0,
                "maximum_violation_magnitude": float(np.max(violations)) if violations else 0.0,
                "mean_prediction_at_time_zero": float(np.mean(time_zero)),
                "minimum_prediction_at_time_zero": float(np.min(time_zero)),
                "maximum_prediction_at_time_zero": float(np.max(time_zero)),
                "mean_maximum_adjacent_grid_step": float(np.mean(maximum_steps)),
                "maximum_adjacent_grid_step": float(np.max(maximum_steps)),
            }
        )
    return pd.DataFrame(rows)


def _first_crossing_time(time: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """Interpolate the first observed threshold crossing without extrapolation."""
    order = np.argsort(time, kind="mergesort")
    time = np.asarray(time, dtype=float)[order]
    values = np.asarray(values, dtype=float)[order]
    crossing = np.flatnonzero(values >= threshold)
    if len(crossing) == 0:
        return float("nan")
    index = int(crossing[0])
    if index == 0 or values[index] == values[index - 1]:
        return float(time[index])
    fraction = (threshold - values[index - 1]) / (values[index] - values[index - 1])
    return float(time[index - 1] + fraction * (time[index] - time[index - 1]))


def curve_trajectory_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate curve-equal trajectory, AUC, final-time, and threshold metrics."""
    rows = []
    group_columns = [column for column in ["task", "model", "study_id", "curve_id"] if column in predictions]
    for key, curve in predictions.groupby(group_columns, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        identity = dict(zip(group_columns, key))
        collapsed = (
            curve.groupby("degradation_time_days", as_index=False)[["y_true", "y_pred"]]
            .mean()
            .sort_values("degradation_time_days")
        )
        time = collapsed.degradation_time_days.to_numpy(dtype=float)
        observed = collapsed.y_true.to_numpy(dtype=float)
        predicted = collapsed.y_pred.to_numpy(dtype=float)
        error = np.abs(observed - predicted)
        if len(time) >= 2 and time[-1] > time[0]:
            integrated_error = float(np.trapz(error, time))
            normalized_integrated_error = integrated_error / float(time[-1] - time[0])
            observed_auc = float(np.trapz(observed, time))
            predicted_auc = float(np.trapz(predicted, time))
            auc_absolute_error = abs(observed_auc - predicted_auc)
            if np.std(observed) > 0 and np.std(predicted) > 0:
                trajectory_correlation = float(np.corrcoef(observed, predicted)[0, 1])
            else:
                trajectory_correlation = float("nan")
        else:
            integrated_error = float("nan")
            normalized_integrated_error = float("nan")
            observed_auc = float("nan")
            predicted_auc = float("nan")
            auc_absolute_error = float("nan")
            trajectory_correlation = float("nan")
        row = {
            **identity,
            "n_observations": len(curve),
            "n_unique_times": len(time),
            "curve_mae": float(error.mean()),
            "curve_rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
            "integrated_absolute_error_pct_days": integrated_error,
            "time_normalized_integrated_absolute_error_pct": normalized_integrated_error,
            "observed_auc_pct_days": observed_auc,
            "predicted_auc_pct_days": predicted_auc,
            "auc_absolute_error_pct_days": auc_absolute_error,
            "trajectory_correlation": trajectory_correlation,
            "final_time_days": float(time[-1]),
            "final_time_observed_pct": float(observed[-1]),
            "final_time_predicted_pct": float(predicted[-1]),
            "final_time_absolute_error_pct": float(abs(observed[-1] - predicted[-1])),
        }
        for threshold in (10.0, 25.0, 50.0):
            observed_time = _first_crossing_time(time, observed, threshold)
            predicted_time = _first_crossing_time(time, predicted, threshold)
            row[f"observed_t{int(threshold)}_days"] = observed_time
            row[f"predicted_t{int(threshold)}_days"] = predicted_time
            row[f"t{int(threshold)}_absolute_error_days"] = (
                abs(observed_time - predicted_time)
                if np.isfinite(observed_time) and np.isfinite(predicted_time)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)
