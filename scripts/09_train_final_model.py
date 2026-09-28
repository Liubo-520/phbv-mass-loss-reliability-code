"""Tune the selected framework on grouped CV and export the deployable model."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.data_io import ROOT
from src.evaluation import summarize_prediction_table
from src.models import build_pipeline, sample_weights, xgboost_variant_spec
from src.preprocessing import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


FINAL_VARIANT = "bounded_monotonic_study_weighted"


def suggest_params(trial: optuna.Trial) -> dict[str, object]:
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


def grouped_predictions(table: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
    splitter = GroupKFold(n_splits=4)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(table, groups=table.study_id_normalized)
    ):
        train = table.iloc[train_idx]
        test = table.iloc[test_idx]
        pipeline = build_pipeline(
            xgboost_variant_spec(
                params,
                bounded=True,
                monotonic=True,
                name=FINAL_VARIANT,
            )
        )
        pipeline.fit(
            train[MODEL_FEATURES],
            train.weight_loss,
            model__sample_weight=sample_weights(train, "study"),
        )
        prediction = pipeline.predict(test[MODEL_FEATURES])
        for (_, row), y_pred in zip(test.iterrows(), prediction):
            rows.append(
                {
                    "task": "final_full_data_group_cv",
                    "fold": f"fold_{fold}",
                    "model": FINAL_VARIANT,
                    "row_id": row.row_id,
                    "study_id": str(row.study_id_normalized),
                    "curve_id": row.curve_id,
                    "curve_position": int(row.curve_position),
                    "degradation_time_days": float(row.degradation_time_days),
                    "degradation_environment": row.degradation_environment,
                    "y_true": float(row.weight_loss),
                    "y_pred": float(y_pred),
                }
            )
    return pd.DataFrame(rows)


def main(trials: int = 12) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    table = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv",
        dtype={"study_id_normalized": str, "instance_normalized": str},
    )
    splitter = GroupKFold(n_splits=4)
    split_indices = list(splitter.split(table, groups=table.study_id_normalized))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        study_errors = []
        for train_idx, validation_idx in split_indices:
            train = table.iloc[train_idx]
            validation = table.iloc[validation_idx]
            pipeline = build_pipeline(
                xgboost_variant_spec(
                    params,
                    bounded=True,
                    monotonic=True,
                    name=FINAL_VARIANT,
                )
            )
            pipeline.fit(
                train[MODEL_FEATURES],
                train.weight_loss,
                model__sample_weight=sample_weights(train, "study"),
            )
            prediction = pipeline.predict(validation[MODEL_FEATURES])
            fold = pd.DataFrame(
                {
                    "study": validation.study_id_normalized.to_numpy(),
                    "absolute_error": np.abs(validation.weight_loss.to_numpy() - prediction),
                }
            )
            study_errors.extend(fold.groupby("study").absolute_error.mean().tolist())
        return float(np.mean(study_errors))

    tuning_dir = ROOT / "results" / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(tuning_dir / 'final_model_raw_time_missingness_v2.db').as_posix()}"
    study = optuna.create_study(
        study_name="final_raw_time_missingness_v2_bounded_monotonic_xgboost",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    if completed < trials:
        study.optimize(objective, n_trials=trials - completed)

    predictions = grouped_predictions(table, study.best_params)
    metrics, curve_metrics, study_metrics = summarize_prediction_table(predictions)
    prediction_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"
    predictions.to_csv(prediction_dir / "final_model_groupcv_predictions.csv", index=False)
    metrics.to_csv(metric_dir / "final_model_groupcv_metrics.csv", index=False)
    curve_metrics.to_csv(metric_dir / "final_model_groupcv_curve_metrics.csv", index=False)
    study_metrics.to_csv(metric_dir / "final_model_groupcv_study_metrics.csv", index=False)
    study.trials_dataframe().to_csv(tuning_dir / "final_model_trials.csv", index=False)

    pipeline = build_pipeline(
        xgboost_variant_spec(
            study.best_params,
            bounded=True,
            monotonic=True,
            name=FINAL_VARIANT,
        )
    )
    pipeline.fit(
        table[MODEL_FEATURES],
        table.weight_loss,
        model__sample_weight=sample_weights(table, "study"),
    )
    app_dir = ROOT / "app"
    app_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, app_dir / "model.joblib", compress=3)

    metadata = {
        "model_name": FINAL_VARIANT,
        "endpoint": "degradation-induced weight loss / physical mass-loss trajectory (%)",
        "not_endpoint": "ultimate biodegradation or mineralization",
        "training_rows": len(table),
        "training_curves": int(table.curve_id.nunique()),
        "training_studies": int(table.study_id_normalized.nunique()),
        "features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "analysis_time": {
            "primary": "raw-decimal time reconstructed from the paired source workbook",
            "sensitivity": "released time truncated toward zero",
        },
        "numeric_ranges": {
            column: {"minimum": float(table[column].min()), "maximum": float(table[column].max())}
            for column in NUMERIC_FEATURES
        },
        "categorical_levels": {
            column: sorted(table[column].astype(str).unique().tolist())
            for column in CATEGORICAL_FEATURES
        },
        "target_transform": {
            "type": "bounded_logit",
            "epsilon": 1e-3,
            "limits": [0, 100],
            "training_target_clip_count": int((table.weight_loss > 100).sum()),
            "prediction_output_clipping": "0-100%",
        },
        "monotonic_constraint": {"feature": "degradation_time_days", "direction": "increasing"},
        "sample_weighting": "study_balanced",
        "best_parameters": study.best_params,
        "grouped_cv_macro_study_mae_objective": float(study.best_value),
        "random_seed": 42,
        "model_export_id": "phbv-bm-xgb-2026-07-27-v2",
        "release_date": "2026-07-27",
        "training_population": (
            "Released additive-enriched PHBV modelling corpus; not a probability sample "
            "of PHBV experiments."
        ),
        "missingness_handling": {
            "restored_original_temperature_missing_rows": 210,
            "temperature_missing_studies": 3,
            "numeric_imputation": "training-fold median only",
            "indicators": [
                "temperature reporting",
                "additive concentration reporting",
                "ambiguous named-additive zero",
            ],
            "remaining_inherited_processing": ["source-level category coarsening"],
        },
        "interval_method": (
            "Post-selection study-grouped cross-validation residual diagnostic; "
            "residual-producing folds also informed hyperparameter selection."
        ),
        "intended_use_limits": (
            "Research hypothesis exploration and experiment prioritisation only; not for "
            "certification, regulatory or safety decisions, causal inference, or replacement "
            "of laboratory testing. Marginal checks do not validate joint scenarios."
        ),
        "citation": {
            "source_corpus": "Kotzabasaki et al. (2026), Polymers 18, 897",
            "source_doi": "10.3390/polym18070897",
            "companion_manuscript": "Bibliographic record pending author submission/publication.",
        },
        "license_scope": (
            "MIT covers project-authored interface code and packaging only; source article, "
            "workbook, CSV, and notebooks retain their original rights and are not uploaded."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "optuna": optuna.__version__,
        },
        "limitations": [
            "The effective validation sample contains only 16 source studies.",
            "Leave-one-study-out errors are large and heterogeneous.",
            "Post-selection grouped residual intervals are often close to the full 0-100% range.",
            "Marginal input checks do not validate joint scenarios.",
            "The corpus is additive-enriched and original temperature/additive reporting is incomplete.",
            "Predictions are associations from literature-derived data, not causal estimates.",
        ],
    }
    (app_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    main(args.trials)
