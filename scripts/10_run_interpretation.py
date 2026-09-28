"""Generate SHAP, held-out grouped permutation, and counterfactual interpretations."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error

from src.data_io import ROOT
from src.models import build_pipeline, sample_weights, xgboost_variant_spec
from src.preprocessing import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


FINAL_VARIANT = "bounded_monotonic_study_weighted"


def original_feature(encoded_name: str) -> str:
    """Map a transformed numeric/OHE feature back to its original input group."""
    candidates = [
        feature
        for feature in MODEL_FEATURES
        if encoded_name == feature or encoded_name.startswith(feature + "_")
    ]
    if not candidates:
        raise ValueError(f"Cannot map encoded feature {encoded_name!r}")
    return max(candidates, key=len)


def grouped_permutation_importance(table: pd.DataFrame, repeats: int = 5) -> pd.DataFrame:
    """Permute original held-out features, preserving static values within curves."""
    selected = pd.read_csv(
        ROOT / "results" / "tuning" / "selected_hyperparameters.csv",
        dtype={"outer_test_study": str},
    )
    selected = selected.loc[selected.model.eq("xgboost")].set_index("outer_test_study")
    rows = []
    for study_index, outer_study in enumerate(sorted(table.study_id_normalized.unique(), key=str)):
        train = table.loc[table.study_id_normalized.ne(outer_study)].copy()
        test = table.loc[table.study_id_normalized.eq(outer_study)].copy()
        params = json.loads(selected.loc[outer_study, "parameters_json"])
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
        baseline_prediction = pipeline.predict(test[MODEL_FEATURES])
        baseline_mae = mean_absolute_error(test.weight_loss, baseline_prediction)
        curve_order = test.curve_id.drop_duplicates().tolist()
        for feature_index, feature in enumerate(MODEL_FEATURES):
            repeat_changes = []
            for repeat in range(repeats):
                rng = np.random.default_rng(42 + study_index * 10_000 + feature_index * 100 + repeat)
                permuted = test.copy()
                if feature == "degradation_time_days":
                    permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                else:
                    curve_values = [
                        test.loc[test.curve_id.eq(curve_id), feature].iloc[0]
                        for curve_id in curve_order
                    ]
                    shuffled = list(np.asarray(curve_values, dtype=object)[rng.permutation(len(curve_values))])
                    mapping = dict(zip(curve_order, shuffled))
                    permuted[feature] = permuted.curve_id.map(mapping)
                permuted_prediction = pipeline.predict(permuted[MODEL_FEATURES])
                repeat_changes.append(
                    mean_absolute_error(test.weight_loss, permuted_prediction) - baseline_mae
                )
            rows.append(
                {
                    "outer_test_study": outer_study,
                    "feature": feature,
                    "n_curves_in_test_study": len(curve_order),
                    "baseline_mae": baseline_mae,
                    "permutation_mae_increase_mean": float(np.mean(repeat_changes)),
                    "permutation_mae_increase_sd": float(np.std(repeat_changes, ddof=1)),
                    "repeats": repeats,
                }
            )
        print(f"completed grouped permutation study {outer_study}", flush=True)
    return pd.DataFrame(rows)


def select_reference_curve(table: pd.DataFrame) -> pd.Series:
    """Select an observed curve nearest numeric medians and categorical modes."""
    curves = table.sort_values("curve_position").groupby("curve_id", as_index=False).first()
    numeric = curves[NUMERIC_FEATURES[1:]]
    scale = numeric.std().replace(0, 1)
    numeric_distance = (((numeric - numeric.median()) / scale) ** 2).mean(axis=1)
    categorical_distance = np.zeros(len(curves))
    for feature in CATEGORICAL_FEATURES:
        categorical_distance += curves[feature].ne(curves[feature].mode().iloc[0]).to_numpy()
    score = numeric_distance.to_numpy() + categorical_distance / len(CATEGORICAL_FEATURES)
    return curves.iloc[int(np.argmin(score))]


def counterfactual_trajectories(model, table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Generate one-variable-at-a-time associational trajectory scenarios."""
    reference = select_reference_curve(table)
    scenarios: list[tuple[str, str, dict[str, object]]] = [("reference", "observed_reference", {})]
    for environment in sorted(table.degradation_environment.unique()):
        scenarios.append(("environment", str(environment), {"degradation_environment": environment}))
    unique_ratios = table[["adjusted_hb_ratio_formulation_mol", "adjusted_hv_ratio_formulation_mol"]].drop_duplicates()
    for quantile in (0.25, 0.50, 0.75):
        target = unique_ratios.adjusted_hb_ratio_formulation_mol.quantile(quantile)
        pair = unique_ratios.iloc[
            (unique_ratios.adjusted_hb_ratio_formulation_mol - target).abs().argmin()
        ]
        scenarios.append(
            (
                "observed_hb_hv_pair",
                f"HB_quantile_{quantile:.2f}",
                {
                    "adjusted_hb_ratio_formulation_mol": pair.adjusted_hb_ratio_formulation_mol,
                    "adjusted_hv_ratio_formulation_mol": pair.adjusted_hv_ratio_formulation_mol,
                },
            )
        )
    for temperature in [float(table.t_deg.min()), float(table.t_deg.median()), float(table.t_deg.max())]:
        scenarios.append(("temperature", f"{temperature:g}_C", {"t_deg": temperature}))
    scenarios.append(
        (
            "additives",
            "hypothetical_no_additives",
            {
                "additives": "no",
                "additive_type_1": "not_applicable",
                "additive_type_2": "not_applicable",
                "additive_type_3": "not_applicable",
                "additive1_percentage_wt": 0.0,
                "additive2_percentage_wt": 0.0,
                "additive3_percentage_wt": 0.0,
            },
        )
    )

    outputs = []
    time_grid = np.linspace(0, 360, 181)
    for scenario_type, label, changes in scenarios:
        frame = pd.DataFrame([reference[MODEL_FEATURES].to_dict()] * len(time_grid))
        frame["degradation_time_days"] = time_grid
        for feature, value in changes.items():
            frame[feature] = value
        prediction = model.predict(frame[MODEL_FEATURES])
        frame.insert(0, "prediction_weight_loss_pct", prediction)
        frame.insert(0, "scenario_label", label)
        frame.insert(0, "scenario_type", scenario_type)
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True), str(reference.curve_id)


def main() -> None:
    table = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv",
        dtype={"study_id_normalized": str, "instance_normalized": str},
    )
    model = joblib.load(ROOT / "app" / "model.joblib")
    preprocessor = model.named_steps["preprocessor"]
    bounded = model.named_steps["model"]
    xgb_model = bounded.estimator_.estimator_

    sample = table.sample(n=min(500, len(table)), random_state=42).sort_index()
    transformed = preprocessor.transform(sample[MODEL_FEATURES])
    feature_names = preprocessor.get_feature_names_out().tolist()
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = np.asarray(explainer.shap_values(transformed))

    encoded_importance = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "original_feature": [original_feature(name) for name in feature_names],
            "mean_absolute_shap_logit_scale": np.abs(shap_values).mean(axis=0),
            "mean_shap_logit_scale": shap_values.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap_logit_scale", ascending=False)
    grouped_importance = (
        encoded_importance.groupby("original_feature", as_index=False)
        .agg(
            mean_absolute_shap_logit_scale=("mean_absolute_shap_logit_scale", "sum"),
            mean_shap_logit_scale=("mean_shap_logit_scale", "sum"),
            encoded_columns=("encoded_feature", "count"),
        )
        .sort_values("mean_absolute_shap_logit_scale", ascending=False)
    )

    shap_long = pd.DataFrame(shap_values, columns=feature_names)
    shap_long.insert(0, "row_id", sample.row_id.to_numpy())
    interaction_sample = transformed[: min(200, len(transformed))]
    interaction_values = np.asarray(explainer.shap_interaction_values(interaction_sample))
    grouped_interactions: dict[tuple[str, str], float] = defaultdict(float)
    for left in range(len(feature_names)):
        for right in range(left + 1, len(feature_names)):
            pair = tuple(sorted((original_feature(feature_names[left]), original_feature(feature_names[right]))))
            if pair[0] == pair[1]:
                continue
            grouped_interactions[pair] += float(np.abs(interaction_values[:, left, right]).mean())
    interaction_table = pd.DataFrame(
        [
            {
                "feature_1": pair[0],
                "feature_2": pair[1],
                "mean_absolute_shap_interaction_logit_scale": value,
            }
            for pair, value in grouped_interactions.items()
        ]
    ).sort_values("mean_absolute_shap_interaction_logit_scale", ascending=False)

    permutation = grouped_permutation_importance(table)
    permutation_summary = (
        permutation.groupby("feature", as_index=False)
        .agg(
            mean_study_equal_mae_increase=("permutation_mae_increase_mean", "mean"),
            median_study_equal_mae_increase=("permutation_mae_increase_mean", "median"),
            study_sd=("permutation_mae_increase_mean", "std"),
            studies=("outer_test_study", "nunique"),
        )
        .sort_values("mean_study_equal_mae_increase", ascending=False)
    )
    counterfactual, reference_curve = counterfactual_trajectories(model, table)

    metric_dir = ROOT / "results" / "metrics"
    encoded_importance.to_csv(metric_dir / "shap_encoded_importance.csv", index=False)
    grouped_importance.to_csv(metric_dir / "shap_grouped_importance.csv", index=False)
    shap_long.to_csv(metric_dir / "shap_values_sample_logit_scale.csv", index=False)
    interaction_table.to_csv(metric_dir / "shap_grouped_interactions.csv", index=False)
    permutation.to_csv(metric_dir / "heldout_grouped_permutation_by_study.csv", index=False)
    permutation_summary.to_csv(metric_dir / "heldout_grouped_permutation_importance.csv", index=False)
    counterfactual.to_csv(metric_dir / "counterfactual_trajectories.csv", index=False)

    notes = [
        "# Interpretation notes",
        "",
        "SHAP values explain the internal bounded-logit target scale of the final full-data XGBoost model. "
        "They are not percentage-point effects and are not causal estimates.",
        "",
        "Held-out permutation importance permutes original feature groups. Static descriptors are shuffled "
        "between complete curves within each held-out study; time is shuffled across held-out rows. The "
        "reported aggregate weights each study equally.",
        "",
        f"Counterfactual trajectories use observed reference curve `{reference_curve}`. HB/HV scenarios use "
        "observed ratio pairs; other scenarios change one descriptor family while holding the reference "
        "values fixed. These are model-response probes, not feasible experimental recipes or causal claims.",
    ]
    (ROOT / "results" / "interpretation_notes.md").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
