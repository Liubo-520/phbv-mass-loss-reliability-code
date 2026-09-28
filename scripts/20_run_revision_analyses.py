"""Run the reviewer-requested transportability and sensitivity analyses.

All reported summaries are calculated from prediction or row-level diagnostic
tables saved by this script.  Original files under ``origin_data`` are read only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.data_io import ROOT
from src.evaluation import point_metrics, summarize_prediction_table
from src.models import ModelSpec, build_pipeline, sample_weights, xgboost_variant_spec
from src.preprocessing import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


PRIMARY_MODEL = "bounded_monotonic_study_weighted"
FOCAL_STUDIES = [9, 19, 12, 14]


def _load_table() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv",
        dtype={"study_id_normalized": str, "instance_normalized": str},
    )


def _prediction_rows(
    test: pd.DataFrame,
    prediction: np.ndarray,
    *,
    task: str,
    model: str,
    fold: str,
    repeat: int | None = None,
    split_seed: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (_, row), y_pred in zip(test.iterrows(), prediction):
        output: dict[str, object] = {
            "task": task,
            "fold": fold,
            "model": model,
            "row_id": row.row_id,
            "study_id": str(row.study_id_normalized),
            "curve_id": row.curve_id,
            "curve_position": int(row.curve_position),
            "degradation_time_days": float(row.degradation_time_days),
            "degradation_environment": row.degradation_environment,
            "y_true": float(row.weight_loss),
            "y_pred": float(y_pred),
        }
        if repeat is not None:
            output["repeat"] = repeat
        if split_seed is not None:
            output["split_seed"] = split_seed
        rows.append(output)
    return rows


def _repeat_metrics_from_saved_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path, dtype={"study_id": str})
    rows = []
    for (task, repeat, seed), group in predictions.groupby(
        ["task", "repeat", "split_seed"], sort=True
    ):
        metrics = point_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())
        curve_mae = group.assign(ae=(group.y_true - group.y_pred).abs()).groupby("curve_id").ae.mean()
        study_mae = group.assign(ae=(group.y_true - group.y_pred).abs()).groupby("study_id").ae.mean()
        rows.append(
            {
                "task": task,
                "repeat": int(repeat),
                "split_seed": int(seed),
                "n_test": len(group),
                "n_test_curves": group.curve_id.nunique(),
                "n_test_studies": group.study_id.nunique(),
                **metrics,
                "macro_curve_mae": float(curve_mae.mean()),
                "macro_study_mae": float(study_mae.mean()),
            }
        )
    return pd.DataFrame(rows)


def run_repeated_splits(table: pd.DataFrame, repeats: int) -> None:
    prediction_path = ROOT / "results" / "predictions" / "repeated_split_predictions.csv"
    # These are the exact fixed-screening hyperparameters.  One thread is used
    # because OpenMP scheduling dominates runtime for this small table; n_jobs
    # does not alter the statistical specification.
    def fixed_factory():
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
        )

    fixed_xgboost = ModelSpec("xgboost", fixed_factory)
    all_indices = np.arange(len(table))
    rows: list[dict[str, object]] = []

    for repeat in range(1, repeats + 1):
        seed = 41 + repeat
        train_idx, test_idx = train_test_split(
            all_indices, test_size=0.20, random_state=seed, shuffle=True
        )
        train = table.iloc[train_idx]
        test = table.iloc[test_idx]
        pipeline = build_pipeline(fixed_xgboost)
        pipeline.fit(train[MODEL_FEATURES], train.weight_loss)
        rows.extend(
            _prediction_rows(
                test,
                pipeline.predict(test[MODEL_FEATURES]),
                task="repeated_random_row",
                model="fixed_xgboost",
                fold=f"repeat_{repeat:02d}",
                repeat=repeat,
                split_seed=seed,
            )
        )

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        curve_train_idx, curve_test_idx = next(
            splitter.split(table, groups=table.curve_id.to_numpy())
        )
        curve_train = table.iloc[curve_train_idx]
        curve_test = table.iloc[curve_test_idx]
        if set(curve_train.curve_id) & set(curve_test.curve_id):
            raise AssertionError("Curve overlap in repeated grouped split")
        pipeline = build_pipeline(fixed_xgboost)
        pipeline.fit(curve_train[MODEL_FEATURES], curve_train.weight_loss)
        rows.extend(
            _prediction_rows(
                curve_test,
                pipeline.predict(curve_test[MODEL_FEATURES]),
                task="repeated_curve_grouped",
                model="fixed_xgboost",
                fold=f"repeat_{repeat:02d}",
                repeat=repeat,
                split_seed=seed,
            )
        )
        print(f"completed repeated splits {repeat}/{repeats}", flush=True)

    pd.DataFrame(rows).to_csv(prediction_path, index=False)
    _repeat_metrics_from_saved_predictions(prediction_path).to_csv(
        ROOT / "results" / "metrics" / "repeated_split_metrics.csv", index=False
    )


def _weight_diagnostics(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    study_rows = []
    for held_out in sorted(table.study_id_normalized.unique(), key=int):
        train = table.loc[table.study_id_normalized.ne(held_out)].copy()
        for strategy in ["study_only", "study_curve", "study_curve_capped95"]:
            weights = sample_weights(train, strategy)
            q95_uncapped = float(np.quantile(sample_weights(train, "study_curve"), 0.95))
            fold_rows.append(
                {
                    "held_out_study": held_out,
                    "strategy": strategy,
                    "n_training_rows": len(train),
                    "minimum": float(np.min(weights)),
                    "q01": float(np.quantile(weights, 0.01)),
                    "q05": float(np.quantile(weights, 0.05)),
                    "median": float(np.median(weights)),
                    "q95": float(np.quantile(weights, 0.95)),
                    "q99": float(np.quantile(weights, 0.99)),
                    "maximum": float(np.max(weights)),
                    "capping_threshold_before_renormalization": (
                        q95_uncapped if strategy == "study_curve_capped95" else np.nan
                    ),
                    "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
                }
            )
            weighted = train[["study_id_normalized", "curve_id"]].copy()
            weighted["weight"] = weights
            for study_id, group in weighted.groupby("study_id_normalized", sort=False):
                study_rows.append(
                    {
                        "held_out_study": held_out,
                        "strategy": strategy,
                        "training_study": study_id,
                        "n_rows": len(group),
                        "n_curves": group.curve_id.nunique(),
                        "study_total_weight": float(group.weight.sum()),
                        "minimum_row_weight": float(group.weight.min()),
                        "maximum_row_weight": float(group.weight.max()),
                    }
                )
    return pd.DataFrame(fold_rows), pd.DataFrame(study_rows)


def run_weight_and_composition_sensitivity(table: pd.DataFrame) -> None:
    selected = pd.read_csv(
        ROOT / "results" / "tuning" / "selected_hyperparameters.csv",
        dtype={"outer_test_study": str},
    )
    selected = selected.loc[selected.model.eq("xgboost")].set_index("outer_test_study")

    existing = pd.read_csv(
        ROOT / "results" / "predictions" / "framework_ablation_predictions.csv",
        dtype={"study_id": str},
    )
    existing = existing.loc[
        existing.model.isin([PRIMARY_MODEL, "bounded_monotonic_row"])
    ].copy()
    existing["task"] = "revision_sensitivity_loso"
    existing["model"] = existing.model.replace(
        {
            PRIMARY_MODEL: "bounded_monotonic_study_curve",
            "bounded_monotonic_row": "bounded_monotonic_row",
        }
    )
    prediction_frames = [existing]

    variants = [
        ("bounded_monotonic_study_only", "study_only", None),
        ("bounded_monotonic_study_curve_capped95", "study_curve_capped95", None),
        ("bounded_monotonic_hv_only", "study_curve", "adjusted_hb_ratio_formulation_mol"),
        ("bounded_monotonic_hb_only", "study_curve", "adjusted_hv_ratio_formulation_mol"),
    ]
    for held_out in sorted(table.study_id_normalized.unique(), key=int):
        train = table.loc[table.study_id_normalized.ne(held_out)].copy()
        test = table.loc[table.study_id_normalized.eq(held_out)].copy()
        params = json.loads(selected.loc[held_out, "parameters_json"])
        params["n_jobs"] = 1
        for model_name, strategy, dropped_feature in variants:
            numeric = [feature for feature in NUMERIC_FEATURES if feature != dropped_feature]
            features = numeric + CATEGORICAL_FEATURES
            spec = xgboost_variant_spec(
                params,
                bounded=True,
                monotonic=True,
                epsilon=1e-3,
                name=model_name,
            )
            pipeline = build_pipeline(spec, numeric_features=numeric)
            pipeline.fit(
                train[features],
                train.weight_loss,
                model__sample_weight=sample_weights(train, strategy),
            )
            prediction_frames.append(
                pd.DataFrame(
                    _prediction_rows(
                        test,
                        pipeline.predict(test[features]),
                        task="revision_sensitivity_loso",
                        model=model_name,
                        fold=f"study_{held_out}",
                    )
                )
            )
        print(f"completed revision sensitivities for study {held_out}", flush=True)

    prediction_path = ROOT / "results" / "predictions" / "revision_sensitivity_predictions.csv"
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(prediction_path, index=False)
    saved = pd.read_csv(prediction_path, dtype={"study_id": str})
    metrics, curve_metrics, study_metrics = summarize_prediction_table(saved)
    metrics.to_csv(ROOT / "results" / "metrics" / "revision_sensitivity_metrics.csv", index=False)
    curve_metrics.to_csv(
        ROOT / "results" / "metrics" / "revision_sensitivity_curve_metrics.csv", index=False
    )
    study_metrics.to_csv(
        ROOT / "results" / "metrics" / "revision_sensitivity_study_metrics.csv", index=False
    )

    weight_folds, weight_studies = _weight_diagnostics(table)
    weight_folds.to_csv(
        ROOT / "results" / "metrics" / "weight_distribution_by_fold.csv", index=False
    )
    weight_studies.to_csv(
        ROOT / "results" / "metrics" / "weight_by_study_summary.csv", index=False
    )

    hb = table.adjusted_hb_ratio_formulation_mol.to_numpy(float)
    hv = table.adjusted_hv_ratio_formulation_mol.to_numpy(float)
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "pearson_hb_hv", "value": float(np.corrcoef(hb, hv)[0, 1])},
            {"diagnostic": "spearman_hb_hv", "value": float(spearmanr(hb, hv).statistic)},
            {"diagnostic": "hb_plus_hv_min", "value": float(np.min(hb + hv))},
            {"diagnostic": "hb_plus_hv_median", "value": float(np.median(hb + hv))},
            {"diagnostic": "hb_plus_hv_max", "value": float(np.max(hb + hv))},
            {"diagnostic": "unique_hb_hv_pairs", "value": float(len(set(zip(hb, hv))))},
        ]
    )
    diagnostics.to_csv(
        ROOT / "results" / "metrics" / "composition_diagnostics.csv", index=False
    )


def _curve_descriptors(group: pd.DataFrame) -> dict[str, float]:
    collapsed = (
        group.groupby("degradation_time_days", as_index=False)[["y_true", "y_pred"]]
        .mean()
        .sort_values("degradation_time_days")
    )
    time = collapsed.degradation_time_days.to_numpy(float)
    y_true = collapsed.y_true.to_numpy(float)
    y_pred = collapsed.y_pred.to_numpy(float)
    terminal_n = min(3, len(collapsed))
    true_slope = float(np.polyfit(time, y_true, 1)[0]) if len(np.unique(time)) >= 2 else np.nan
    pred_slope = float(np.polyfit(time, y_pred, 1)[0]) if len(np.unique(time)) >= 2 else np.nan
    return {
        "observed_initial": float(y_true[0]),
        "predicted_initial": float(y_pred[0]),
        "observed_terminal": float(np.mean(y_true[-terminal_n:])),
        "predicted_terminal": float(np.mean(y_pred[-terminal_n:])),
        "observed_linear_slope": true_slope,
        "predicted_linear_slope": pred_slope,
    }


def build_study_shift_summary(table: pd.DataFrame) -> None:
    primary = pd.read_csv(
        ROOT / "results" / "predictions" / "framework_ablation_predictions.csv",
        dtype={"study_id": str},
    )
    primary = primary.loc[primary.model.eq(PRIMARY_MODEL)].copy()
    applicability = pd.read_csv(
        ROOT / "results" / "metrics" / "applicability_indicators.csv",
        dtype={"study_id": str},
    )
    merged = primary.merge(
        table[["row_id", "study_id_normalized", *MODEL_FEATURES]],
        on="row_id",
        validate="one_to_one",
        suffixes=("", "_source"),
    ).merge(
        applicability.drop(columns=["study_id", "curve_id", "absolute_error"]),
        on="row_id",
        validate="one_to_one",
    )

    shift_rows = []
    row_diagnostics = []
    for held_out in sorted(table.study_id_normalized.unique(), key=int):
        train = table.loc[table.study_id_normalized.ne(held_out)]
        test = table.loc[table.study_id_normalized.eq(held_out)]
        numeric_flags: dict[str, np.ndarray] = {}
        for feature in NUMERIC_FEATURES:
            lower = float(train[feature].min())
            upper = float(train[feature].max())
            numeric_flags[feature] = (
                test[feature].notna()
                & ((test[feature] < lower) | (test[feature] > upper))
            ).to_numpy()
        unseen_flags: dict[str, np.ndarray] = {}
        for feature in CATEGORICAL_FEATURES:
            known = set(train[feature].dropna().astype(str))
            unseen_flags[feature] = (
                test[feature].notna() & ~test[feature].astype(str).isin(known)
            ).to_numpy()
        numeric_count = np.sum(np.column_stack(list(numeric_flags.values())), axis=1)
        unseen_count = np.sum(np.column_stack(list(unseen_flags.values())), axis=1)
        for row_id, n_count, c_count in zip(test.row_id, numeric_count, unseen_count):
            row_diagnostics.append(
                {
                    "row_id": row_id,
                    "study_id": held_out,
                    "numeric_range_violation_count_recomputed": int(n_count),
                    "unseen_category_count_recomputed": int(c_count),
                }
            )

        group = merged.loc[merged.study_id.eq(held_out)].copy()
        group["absolute_error"] = (group.y_true - group.y_pred).abs()
        trajectory = pd.DataFrame(
            [_curve_descriptors(curve) for _, curve in group.groupby("curve_id", sort=False)]
        )
        study_metrics = point_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())
        numeric_features_outside = [name for name, flag in numeric_flags.items() if flag.any()]
        unseen_features = [name for name, flag in unseen_flags.items() if flag.any()]
        shift_rows.append(
            {
                "study_id": held_out,
                "n": len(group),
                "n_curves": group.curve_id.nunique(),
                **study_metrics,
                "response_range": float(group.y_true.max() - group.y_true.min()),
                "normalized_mae_by_response_range": (
                    float(study_metrics["mae"] / (group.y_true.max() - group.y_true.min()))
                    if group.y_true.max() - group.y_true.min() >= 5
                    else np.nan
                ),
                "target_mean": float(group.y_true.mean()),
                "target_max": float(group.y_true.max()),
                "prediction_mean": float(group.y_pred.mean()),
                "time_min": float(group.degradation_time_days.min()),
                "time_max": float(group.degradation_time_days.max()),
                "knn_distance_mean": float(group.knn_distance_mean.mean()),
                "knn_distance_max": float(group.knn_distance_max.max()),
                "numeric_range_violation_mean": float(numeric_count.mean()),
                "numeric_range_violation_max": int(numeric_count.max()),
                "numeric_features_outside_training_range_count": len(numeric_features_outside),
                "numeric_features_outside_training_range": "; ".join(numeric_features_outside),
                "numeric_outside_row_fraction": float(np.mean(numeric_count > 0)),
                "unseen_category_mean": float(unseen_count.mean()),
                "unseen_category_max": int(unseen_count.max()),
                "unseen_category_feature_count": len(unseen_features),
                "unseen_category_features": "; ".join(unseen_features),
                "unseen_category_row_fraction": float(np.mean(unseen_count > 0)),
                "environment": "; ".join(sorted(group.degradation_environment.dropna().unique())),
                "experimental_condition": "; ".join(sorted(group.degradation_condition.dropna().unique())),
                "experimental_scale": "; ".join(sorted(group.experimental_scale.dropna().unique())),
                "observed_initial_mean": float(trajectory.observed_initial.mean()),
                "predicted_initial_mean": float(trajectory.predicted_initial.mean()),
                "observed_terminal_mean": float(trajectory.observed_terminal.mean()),
                "predicted_terminal_mean": float(trajectory.predicted_terminal.mean()),
                "observed_linear_slope_mean": float(trajectory.observed_linear_slope.mean()),
                "predicted_linear_slope_mean": float(trajectory.predicted_linear_slope.mean()),
            }
        )

    row_path = ROOT / "results" / "metrics" / "study_shift_row_diagnostics.csv"
    pd.DataFrame(row_diagnostics).to_csv(row_path, index=False)
    diagnostics = pd.read_csv(row_path, dtype={"study_id": str})
    check = applicability[["row_id", "numeric_range_violation_count", "unseen_category_count"]].merge(
        diagnostics, on="row_id", validate="one_to_one"
    )
    if not (
        check.numeric_range_violation_count.eq(
            check.numeric_range_violation_count_recomputed
        ).all()
        and check.unseen_category_count.eq(check.unseen_category_count_recomputed).all()
    ):
        raise AssertionError("Recomputed applicability counts disagree with saved fold-local counts")

    shift = pd.DataFrame(shift_rows)
    shift["r2_for_reporting"] = shift.r2.where(
        shift.n.ge(20) & shift.response_range.ge(5), np.nan
    )
    shift.to_csv(ROOT / "results" / "metrics" / "study_level_shift_summary.csv", index=False)


def build_time_stage_summary() -> None:
    primary = pd.read_csv(
        ROOT / "results" / "predictions" / "framework_ablation_predictions.csv",
        dtype={"study_id": str},
    )
    primary = primary.loc[primary.model.eq(PRIMARY_MODEL)].copy()
    primary["time_stage"] = pd.cut(
        primary.degradation_time_days,
        bins=[-np.inf, 7, 30, 90, np.inf],
        labels=["0-<7 d", "7-<30 d", "30-<90 d", ">=90 d"],
        right=False,
    )
    rows = []
    for stage, group in primary.groupby("time_stage", observed=True, sort=False):
        metrics = point_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())
        study_mae = group.assign(ae=(group.y_true - group.y_pred).abs()).groupby("study_id").ae.mean()
        rows.append(
            {
                "time_stage": str(stage),
                "n": len(group),
                "n_studies": group.study_id.nunique(),
                **metrics,
                "macro_study_mae": float(study_mae.mean()),
            }
        )
    pd.DataFrame(rows).to_csv(
        ROOT / "results" / "metrics" / "time_stage_error.csv", index=False
    )


def _unique_text(series: pd.Series) -> str:
    values = []
    for value in series.dropna().astype(str).str.strip():
        if value and value.lower() != "nan" and value not in values:
            values.append(value)
    return "; ".join(values)


def _reported_range(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "not reported"
    if float(numeric.min()) == float(numeric.max()):
        return f"{numeric.min():g}"
    return f"{numeric.min():g}-{numeric.max():g}"


def build_focal_source_context() -> None:
    workbook = ROOT / "origin_data" / "Weight-loss-model-main" / "scl_weight_loss_data_library_raw_LM.xlsx"
    metadata = pd.read_excel(workbook, sheet_name="metadata")
    environmental = pd.read_excel(workbook, sheet_name="Environmental Properties")
    physicochemical = pd.read_excel(workbook, sheet_name="Physicochemical properties")
    for frame in [metadata, environmental, physicochemical]:
        frame.columns = [str(column).strip() for column in frame.columns]
    metadata["study_id"] = pd.to_numeric(metadata.study_id, errors="coerce")
    environmental["Study_id"] = pd.to_numeric(environmental.Study_id, errors="coerce")
    physicochemical["Study_id"] = pd.to_numeric(physicochemical.Study_id, errors="coerce")

    rows = []
    for study_id in FOCAL_STUDIES:
        meta = metadata.loc[metadata.study_id.eq(study_id)].iloc[0]
        env = environmental.loc[environmental.Study_id.eq(study_id)]
        phys = physicochemical.loc[physicochemical.Study_id.eq(study_id)]
        recorded_omitted = []
        omitted_candidates = {
            "Mw (kDa)": "molecular weight",
            "Density": "density",
            "Void_content (%)": "void content",
            "Size_thickness": "thickness",
            "PHBV _crystallinity (%)": "crystallinity",
            "*Water_absorption_capacity %": "water absorption",
            "**Thickness_swelling (%)": "thickness swelling",
            "Static_water_contact _angle, (WCA) for hydrophilicity (deg)": "water contact angle",
        }
        for column, label in omitted_candidates.items():
            if column in phys and phys[column].notna().any():
                recorded_omitted.append(label)
        for column, label in {
            "Soil_moisture (%)": "soil moisture",
            "Soil_pH": "soil pH",
            "Compost_moisture (%)": "compost moisture",
            "Compost_pH": "compost pH",
            "Salinity_marine (ppt)": "salinity",
        }.items():
            if column in env and env[column].notna().any():
                recorded_omitted.append(label)
        key_missing = []
        for column, label, source in [
            ("Size_thickness", "thickness", phys),
            ("PHBV _crystallinity (%)", "crystallinity", phys),
            ("Mw (kDa)", "molecular weight", phys),
        ]:
            if column not in source or not source[column].notna().any():
                key_missing.append(label)
        rows.append(
            {
                "study_id": study_id,
                "paper_title": str(meta["Paper Title"]),
                "doi": str(meta["DOI"]),
                "environment": _unique_text(env["Degradation_Environment"]),
                "condition": _unique_text(env["Degradation_condition"]),
                "temperature_c": _reported_range(env["T_deg"]),
                "mechanism": _unique_text(env["Degradation_mechanism"]),
                "microbes": _unique_text(env["PHA_degrading_microbes"]),
                "hb_mol_percent_range": _reported_range(
                    phys["Adjusted_HB_ratio_formulation (mol%)"]
                ),
                "hv_mol_percent_range": _reported_range(
                    phys["Adjusted_HV_ratio_formulation (mol%)"]
                ),
                "additive_names": "; ".join(
                    value
                    for value in [
                        _unique_text(phys.get("Additive1_name", pd.Series(dtype=object))),
                        _unique_text(phys.get("Additive2_name", pd.Series(dtype=object))),
                        _unique_text(phys.get("Additive3_name", pd.Series(dtype=object))),
                    ]
                    if value
                ),
                "additive_percentages": "; ".join(
                    value
                    for value in [
                        _unique_text(phys.get("Additive1_percentage (wt%)", pd.Series(dtype=object))),
                        _unique_text(phys.get("Additive2_percentage (wt%)", pd.Series(dtype=object))),
                        _unique_text(phys.get("Additive3_percentage (wt%)", pd.Series(dtype=object))),
                    ]
                    if value
                ),
                "morphology": _unique_text(phys["Sample_shape/Morphology"]),
                "thickness": _unique_text(phys["Size_thickness"]),
                "thickness_units": _unique_text(phys["Size_units"]),
                "source_reported_but_model_omitted": "; ".join(recorded_omitted),
                "key_fields_not_reported_in_workbook": "; ".join(key_missing),
            }
        )
    pd.DataFrame(rows).to_csv(
        ROOT / "results" / "metrics" / "focal_study_source_context.csv", index=False
    )


def main(repeats: int) -> None:
    table = _load_table()
    run_repeated_splits(table, repeats)
    run_weight_and_composition_sensitivity(table)
    build_study_shift_summary(table)
    build_time_stage_summary()
    build_focal_source_context()
    print("Revision analyses completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    main(repeats=args.repeats)
