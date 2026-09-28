"""Generate post-selection grouped residual intervals and applicability indicators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.applicability_domain import applicability_indicators
from src.conformal import interval_score, summarize_intervals, weighted_conformal_quantile
from src.data_io import ROOT
from src.evaluation import summarize_dense_physical_predictions
from src.models import build_pipeline, sample_weights, xgboost_variant_spec
from src.preprocessing import MODEL_FEATURES


COVERAGES = [0.80, 0.90, 0.95]
FINAL_VARIANT = "bounded_monotonic_study_weighted"


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
    inner_all = pd.read_csv(
        ROOT / "splits" / "task_c_inner_grouped.csv",
        dtype={"outer_test_study": str, "study_id": str},
    )
    point_predictions = pd.read_csv(
        ROOT / "results" / "predictions" / "framework_ablation_predictions.csv",
        dtype={"study_id": str},
    )
    point_predictions = point_predictions.loc[point_predictions.model.eq(FINAL_VARIANT)].copy()

    outputs = []
    residual_outputs = []
    quantile_rows = []
    for outer_study in sorted(table.study_id_normalized.unique(), key=str):
        outer_train = table.loc[table.study_id_normalized.ne(outer_study)].copy()
        outer_test = table.loc[table.study_id_normalized.eq(outer_study)].copy()
        params = json.loads(selected.loc[outer_study, "parameters_json"])
        inner = inner_all.loc[inner_all.outer_test_study.eq(outer_study), ["row_id", "inner_validation_fold"]]
        assignment = outer_train[["row_id"]].merge(inner, on="row_id", validate="one_to_one")
        oof_rows = []
        for inner_fold in sorted(assignment.inner_validation_fold.unique()):
            validation_mask = assignment.inner_validation_fold.eq(inner_fold).to_numpy()
            fit = outer_train.loc[~validation_mask]
            validation = outer_train.loc[validation_mask]
            pipeline = build_pipeline(
                xgboost_variant_spec(
                    params,
                    bounded=True,
                    monotonic=True,
                    name=FINAL_VARIANT,
                )
            )
            pipeline.fit(
                fit[MODEL_FEATURES],
                fit["weight_loss"],
                model__sample_weight=sample_weights(fit, "study"),
            )
            prediction = pipeline.predict(validation[MODEL_FEATURES])
            oof_rows.append(
                pd.DataFrame(
                    {
                        "row_id": validation.row_id.to_numpy(),
                        "study_id": validation.study_id_normalized.to_numpy(),
                        "curve_id": validation.curve_id.to_numpy(),
                        "y_true": validation.weight_loss.to_numpy(),
                        "y_pred": prediction,
                        "absolute_residual": np.abs(validation.weight_loss.to_numpy() - prediction),
                    }
                )
            )
        residuals = pd.concat(oof_rows, ignore_index=True)
        weight_lookup = pd.Series(
            sample_weights(outer_train, "study"), index=outer_train.row_id
        )
        residuals["conformal_weight"] = residuals.row_id.map(weight_lookup)
        residuals.insert(0, "outer_test_study", outer_study)
        residual_outputs.append(residuals)

        fold_prediction = point_predictions.loc[point_predictions.study_id.eq(outer_study)].copy()
        if set(fold_prediction.row_id) != set(outer_test.row_id):
            raise ValueError(f"Point-prediction rows do not match held-out study {outer_study}")
        for coverage in COVERAGES:
            suffix = str(int(round(coverage * 100)))
            quantile = weighted_conformal_quantile(
                residuals.absolute_residual.to_numpy(),
                residuals.conformal_weight.to_numpy(),
                coverage,
            )
            fold_prediction[f"lower_{suffix}"] = np.clip(fold_prediction.y_pred - quantile, 0, 100)
            fold_prediction[f"upper_{suffix}"] = np.clip(fold_prediction.y_pred + quantile, 0, 100)
            quantile_rows.append(
                {
                    "outer_test_study": outer_study,
                    "nominal_coverage": coverage,
                    "residual_quantile": quantile,
                    "calibration_rows": len(residuals),
                    "calibration_studies": residuals.study_id.nunique(),
                    "weighting": "study_balanced",
                }
            )
        domain = applicability_indicators(outer_train, outer_test)
        fold_prediction = fold_prediction.merge(domain, on="row_id", validate="one_to_one")
        outputs.append(fold_prediction)
        print(f"completed uncertainty outer study {outer_study}", flush=True)

    predictions = pd.concat(outputs, ignore_index=True)
    predictions["absolute_error"] = (predictions.y_true - predictions.y_pred).abs()
    predictions["interval_width_90"] = predictions.upper_90 - predictions.lower_90
    interval_summary = summarize_intervals(predictions, COVERAGES)

    subgroup_rows = []
    for grouping in ["study_id", "degradation_environment"]:
        for group_id, group in predictions.groupby(grouping):
            for coverage in COVERAGES:
                suffix = str(int(round(coverage * 100)))
                covered = group.y_true.between(group[f"lower_{suffix}"], group[f"upper_{suffix}"])
                subgroup_rows.append(
                    {
                        "grouping": grouping,
                        "group": group_id,
                        "nominal_coverage": coverage,
                        "empirical_coverage": float(covered.mean()),
                        "mean_width": float((group[f"upper_{suffix}"] - group[f"lower_{suffix}"]).mean()),
                        "mean_interval_score": float(
                            interval_score(
                                group.y_true.to_numpy(),
                                group[f"lower_{suffix}"].to_numpy(),
                                group[f"upper_{suffix}"].to_numpy(),
                                coverage,
                            ).mean()
                        ),
                        "n": len(group),
                    }
                )

    correlation_rows = []
    indicators = [
        "knn_distance_mean",
        "knn_distance_max",
        "numeric_range_violation_count",
        "unseen_category_count",
        "interval_width_90",
    ]
    correlation_frames = {
        "point": predictions,
        "curve": predictions.groupby(["study_id", "curve_id"], as_index=False)[
            indicators + ["absolute_error"]
        ].mean(),
        "study": predictions.groupby("study_id", as_index=False)[
            indicators + ["absolute_error"]
        ].mean(),
    }
    for level, correlation_frame in correlation_frames.items():
        for indicator in indicators:
            correlation, p_value = spearmanr(
                correlation_frame[indicator], correlation_frame.absolute_error
            )
            correlation_rows.append(
                {
                    "aggregation_level": level,
                    "indicator": indicator,
                    "spearman_rho_with_absolute_error": correlation,
                    "p_value_descriptive_only": p_value,
                    "n": len(correlation_frame),
                }
            )

    dense = pd.read_csv(ROOT / "results" / "predictions" / "framework_ablation_dense_grid.csv")
    dense_physical = summarize_dense_physical_predictions(dense)
    prediction_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"
    predictions.to_csv(prediction_dir / "final_conformal_predictions.csv", index=False)
    pd.concat(residual_outputs, ignore_index=True).to_csv(
        prediction_dir / "conformal_training_oof_residuals.csv", index=False
    )
    pd.DataFrame(quantile_rows).to_csv(metric_dir / "conformal_quantiles.csv", index=False)
    interval_summary.to_csv(metric_dir / "conformal_interval_metrics.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(metric_dir / "conformal_subgroup_metrics.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(metric_dir / "applicability_correlations.csv", index=False)
    predictions[
        [
            "row_id",
            "study_id",
            "curve_id",
            "knn_distance_mean",
            "knn_distance_max",
            "numeric_range_violation_count",
            "unseen_category_count",
            "interval_width_90",
            "absolute_error",
        ]
    ].to_csv(metric_dir / "applicability_indicators.csv", index=False)
    dense_physical.to_csv(metric_dir / "framework_dense_physical_metrics.csv", index=False)


if __name__ == "__main__":
    main()
