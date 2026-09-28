"""Nested study-grouped tuning for the top three eligible boosting models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.data_io import ROOT
from src.evaluation import summarize_prediction_table
from src.models import build_pipeline, tuned_model_spec
from src.preprocessing import MODEL_FEATURES


ELIGIBLE_MODELS = ["catboost", "xgboost", "hist_gradient_boosting"]


def suggest_params(trial: optuna.Trial, model: str) -> dict[str, object]:
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        }
    if model == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 250, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "depth": trial.suggest_int("depth", 5, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
        }
    if model == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 150, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.10, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 40),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 8.0, log=True),
        }
    raise ValueError(model)


def objective_factory(
    table: pd.DataFrame, inner_assignment: pd.DataFrame, model: str
):
    assignment = table[["row_id"]].merge(
        inner_assignment[["row_id", "inner_validation_fold"]],
        on="row_id",
        validate="one_to_one",
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model)
        held_out = []
        for inner_fold in sorted(assignment.inner_validation_fold.unique()):
            validation_mask = assignment.inner_validation_fold.eq(inner_fold).to_numpy()
            train = table.loc[~validation_mask]
            validation = table.loc[validation_mask]
            pipeline = build_pipeline(tuned_model_spec(model, params))
            pipeline.fit(train[MODEL_FEATURES], train["weight_loss"])
            pred = pipeline.predict(validation[MODEL_FEATURES])
            fold = pd.DataFrame(
                {
                    "study": validation["study_id_normalized"].to_numpy(),
                    "absolute_error": np.abs(validation["weight_loss"].to_numpy() - pred),
                }
            )
            held_out.append(fold)
        residuals = pd.concat(held_out, ignore_index=True)
        return float(residuals.groupby("study")["absolute_error"].mean().mean())

    return objective


def stable_seed(study: str, model_index: int) -> int:
    return 42 + sum(ord(character) for character in study) + 1000 * model_index


def main(trials: int = 12) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    table = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv",
        dtype={"study_id_normalized": str, "instance_normalized": str},
    )
    inner_all = pd.read_csv(
        ROOT / "splits" / "task_c_inner_grouped.csv",
        dtype={"outer_test_study": str, "study_id": str},
    )
    tuning_dir = ROOT / "results" / "tuning"
    prediction_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    # Version the study store so trials from the former truncated-time/global-
    # completion analysis cannot be silently reused after data correction.
    storage = f"sqlite:///{(tuning_dir / 'nested_loso_raw_time_missingness_v2.db').as_posix()}"

    prediction_rows = []
    selected_rows = []
    runtime_rows = []
    for outer_study in sorted(table.study_id_normalized.unique(), key=str):
        outer_train = table.loc[table.study_id_normalized.ne(outer_study)].reset_index(drop=True)
        outer_test = table.loc[table.study_id_normalized.eq(outer_study)].copy()
        inner = inner_all.loc[inner_all.outer_test_study.eq(outer_study)]
        for model_index, model in enumerate(ELIGIBLE_MODELS):
            study_name = f"raw_time_missingness_v2_{model}_outer_{outer_study}"
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=stable_seed(outer_study, model_index)),
            )
            # A command timeout can leave an SQLite-backed trial marked RUNNING.
            # It contains no completed objective and must not remain ambiguous
            # when the deterministic workflow resumes.
            for stale_trial in study.trials:
                if stale_trial.state == optuna.trial.TrialState.RUNNING:
                    study.tell(stale_trial.number, state=optuna.trial.TrialState.FAIL)
            completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
            if completed < trials:
                study.optimize(
                    objective_factory(outer_train, inner, model),
                    n_trials=trials - completed,
                    catch=(ValueError, RuntimeError),
                )
            if not study.best_trials:
                raise RuntimeError(f"No completed tuning trial for {study_name}")

            fit_start = time.perf_counter()
            pipeline = build_pipeline(tuned_model_spec(model, study.best_params))
            pipeline.fit(outer_train[MODEL_FEATURES], outer_train["weight_loss"])
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            prediction = pipeline.predict(outer_test[MODEL_FEATURES])
            predict_seconds = time.perf_counter() - predict_start
            selected_rows.append(
                {
                    "outer_test_study": outer_study,
                    "model": model,
                    "inner_macro_study_mae": study.best_value,
                    "best_trial": study.best_trial.number,
                    "parameters_json": json.dumps(study.best_params, sort_keys=True),
                }
            )
            runtime_rows.append(
                {
                    "outer_test_study": outer_study,
                    "model": model,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
            )
            for (_, row), y_pred in zip(outer_test.iterrows(), prediction):
                prediction_rows.append(
                    {
                        "task": "nested_leave_one_study_out",
                        "fold": f"study_{outer_study}",
                        "model": model,
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
            print(
                f"completed nested LOSO {outer_study}/{model}; "
                f"inner macro-study MAE={study.best_value:.3f}",
                flush=True,
            )

    predictions = pd.DataFrame(prediction_rows)
    metrics, curve_metrics, study_metrics = summarize_prediction_table(predictions)
    predictions.to_csv(prediction_dir / "nested_loso_predictions.csv", index=False)
    metrics.to_csv(metric_dir / "nested_loso_metrics.csv", index=False)
    curve_metrics.to_csv(metric_dir / "nested_loso_curve_metrics.csv", index=False)
    study_metrics.to_csv(metric_dir / "nested_loso_study_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(tuning_dir / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(tuning_dir / "nested_loso_runtime.csv", index=False)
    trial_frames = []
    for study_summary in optuna.study.get_all_study_summaries(storage=storage):
        study = optuna.load_study(study_name=study_summary.study_name, storage=storage)
        trials_frame = study.trials_dataframe()
        trials_frame.insert(0, "study_name", study.study_name)
        trial_frames.append(trials_frame)
    pd.concat(trial_frames, ignore_index=True).to_csv(tuning_dir / "all_trials.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    main(trials=args.trials)
