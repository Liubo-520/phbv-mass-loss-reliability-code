"""Dataset audit and hierarchy summaries."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.preprocessing import MODEL_FEATURES


TRACEABILITY_COLUMNS = {
    "row_id",
    "curve_id",
    "study_id",
    "study_id_normalized",
    "instance",
    "instance_normalized",
    "sample_name",
    "sample_name_target_sheet",
    "sample_name_exact_match",
    "source_cleaned_row",
    "source_row",
    "source_column_position",
    "curve_position",
}
TARGET_COLUMNS = {
    "weight_loss",
    "weight_loss_decimal",
    "weight_loss_cleaned",
    "target_absolute_difference_pct",
    "target_truncation_matches",
}


COLUMN_METADATA = {
    "row_id": ("Stable identifier for one reconstructed observation.", "identifier"),
    "curve_id": ("Stable identifier for one specimen-level time series.", "identifier"),
    "study_id_normalized": ("Normalised source-study identifier used for grouping.", "identifier"),
    "instance_normalized": ("Normalised source curve-instance label.", "identifier"),
    "sample_name": ("Source sample description retained for traceability.", "text"),
    "curve_position": ("Zero-based observation position within the reconstructed curve.", "index"),
    "source_cleaned_row": ("Zero-based row position in the released cleaned CSV.", "index"),
    "source_row": ("Excel row containing the source time/target pair.", "Excel row"),
    "source_column_position": ("Zero-based source column position of the reconstructed curve.", "index"),
    "degradation_time_raw_days": ("Decimal degradation time reconstructed from the raw workbook.", "days"),
    "degradation_time_days": ("Primary raw-decimal degradation time used for modelling.", "days"),
    "degradation_time_cleaned_days": ("Released modelling-table time, truncated toward zero upstream and retained only for sensitivity analysis.", "days"),
    "weight_loss_decimal": ("Decimal cumulative gravimetric mass/weight loss reconstructed from the raw workbook.", "percentage points"),
    "weight_loss_cleaned": ("Released source-table target, truncated toward zero upstream.", "percentage points"),
    "weight_loss": ("Primary raw-decimal response retained for evaluation and audit.", "percentage points"),
    "study_id": ("Original source-study identifier.", "identifier"),
    "instance": ("Original source curve-instance label.", "identifier"),
    "adjusted_hb_ratio_formulation_mol": ("Source-curated adjusted HB formulation ratio; its derivation formula is not documented in the released notebook.", "mol%"),
    "adjusted_hv_ratio_formulation_mol": ("Source-curated adjusted HV formulation ratio; its derivation formula is not documented in the released notebook.", "mol%"),
    "degradation_condition": ("Source-curated oxygen/exposure condition category.", "category"),
    "degradation_mechanism": ("Source-coarsened degradation-mechanism category; see category crosswalk.", "category"),
    "additives": ("Source-curated indicator that the formulation contains an additive.", "category"),
    "t_deg": ("Originally reported degradation temperature; missing values remain missing for fold-local handling.", "degrees Celsius"),
    "temperature_original_c": ("Originally reported degradation temperature reconstructed from the raw workbook.", "degrees Celsius"),
    "t_deg_source_completed": ("Released modelling-table temperature after upstream global environment-mean completion.", "degrees Celsius"),
    "temperature_was_reported": ("Indicator that temperature was reported in the raw workbook.", "boolean"),
    "temperature_was_source_imputed": ("Indicator that the released modelling table contains an upstream-completed temperature for an originally missing value.", "boolean"),
    "degradation_environment": ("Source-coarsened degradation environment; see category crosswalk.", "category"),
    "additive_type_1": ("Source-coarsened type of the first additive; see category crosswalk.", "category"),
    "additive1_percentage_wt": ("First additive concentration with source missingness restored; structural absence is encoded as zero.", "wt%"),
    "additive_type_2": ("Source-coarsened type of the second additive.", "category"),
    "additive2_percentage_wt": ("Second additive concentration with source missingness restored; structural absence is encoded as zero.", "wt%"),
    "additive_type_3": ("Source-coarsened type of the third additive.", "category"),
    "additive3_percentage_wt": ("Third additive concentration with source missingness restored; structural absence is encoded as zero.", "wt%"),
    "sample_shape_morphology": ("Source-coarsened specimen morphology; see category crosswalk.", "category"),
    "pha_degrading_microbes": ("Source-coarsened PHA-degrading microbial category; see category crosswalk.", "category"),
    "experimental_scale": ("Source-curated laboratory or large-scale category.", "category"),
    "sample_name_target_sheet": ("Sample label recovered from the raw target worksheet for linkage audit.", "text"),
    "sample_name_exact_match": ("Whether cleaned-table and raw-target sample labels match exactly.", "boolean"),
    "time_absolute_difference_days": ("Absolute difference between decimal and released degradation time.", "days"),
    "target_absolute_difference_pct": ("Absolute difference between decimal and released mass-loss target.", "percentage points"),
    "time_truncation_matches": ("Whether the released time equals truncation toward zero of the raw decimal time.", "boolean"),
    "target_truncation_matches": ("Whether the released target equals truncation toward zero of the raw decimal target.", "boolean"),
}

DERIVATION_NOTES = {
    "weight_loss": "Copied from weight_loss_decimal after verified positional reconstruction.",
    "weight_loss_decimal": "Positionally reconstructed from paired raw-workbook time and target sheets.",
    "weight_loss_cleaned": "Inherited directly from the released cleaned CSV.",
    "adjusted_hb_ratio_formulation_mol": "Inherited source-curated field; derivation formula is not documented in the released notebook.",
    "adjusted_hv_ratio_formulation_mol": "Inherited source-curated field; derivation formula is not documented in the released notebook.",
    "degradation_time_days": "Copied from degradation_time_raw_days after verified positional reconstruction; this is the primary analysis time.",
    "degradation_time_cleaned_days": "Inherited directly from the released modelling table and retained only for truncation sensitivity analysis.",
    "t_deg": "Copied from temperature_original_c; missing values are imputed within each training fold.",
    "temperature_original_c": "Recovered from the Environmental Properties worksheet before upstream environment-mean completion.",
    "t_deg_source_completed": "Inherited from the released modelling table; 210 rows contain upstream global environment-mean completions.",
    "degradation_mechanism": "Mapped upstream using the rules in results/category_crosswalk.csv.",
    "degradation_environment": "Mapped upstream using the rules in results/category_crosswalk.csv.",
    "sample_shape_morphology": "Mapped upstream using the rules in results/category_crosswalk.csv.",
    "pha_degrading_microbes": "Mapped upstream using the rules in results/category_crosswalk.csv.",
    "additive_type_1": "Mapped upstream using the rules in results/category_crosswalk.csv.",
    "additive1_percentage_wt": "Recovered from the raw workbook; named additives with unreported or ambiguous concentration remain missing, whereas absent additives are structural zero.",
    "additive2_percentage_wt": "Recovered from the raw workbook; named additives with unreported or ambiguous concentration remain missing, whereas absent additives are structural zero.",
    "additive3_percentage_wt": "Recovered from the raw workbook; named additives with unreported or ambiguous concentration remain missing, whereas absent additives are structural zero.",
    "time_absolute_difference_days": "Absolute value of degradation_time_raw_days minus degradation_time_cleaned_days.",
    "target_absolute_difference_pct": "Absolute value of weight_loss_decimal minus weight_loss_cleaned.",
    "time_truncation_matches": "Boolean comparison of released time with truncation of raw decimal time.",
    "target_truncation_matches": "Boolean comparison of released target with truncation of raw decimal target.",
}

MISSINGNESS_NOTES = {
    "t_deg": "Missing means temperature was not reported in the raw workbook; imputation parameters are estimated from training studies only.",
    "temperature_original_c": "Missing means temperature was not reported in the raw workbook.",
    "t_deg_source_completed": "No released-table missingness remains; 210 values are upstream environment-mean completions rather than measurements.",
    "additive1_percentage_wt": "Missing means a named additive concentration was unreported or could not be interpreted as wt%; structural additive absence is zero.",
    "additive2_percentage_wt": "Missing means a named additive concentration was unreported or could not be interpreted as wt%; structural additive absence is zero.",
    "additive3_percentage_wt": "Missing means a named additive concentration was unreported or could not be interpreted as wt%; structural additive absence is zero.",
    "sample_name_target_sheet": "Missing means the raw target sheet did not supply a recoverable sample label.",
    "sample_name_exact_match": "Missing would mean that an exact label comparison could not be formed.",
}


def category_crosswalk() -> pd.DataFrame:
    """Return categorical coarsening rules recovered from the source notebook."""
    mappings = {
        "degradation_mechanism": {
            "microbial_biodegradation": "microbial_enzymatic",
            "enzymemediated_microbial_biodegradation": "microbial_enzymatic",
            "hydrolysisassisted_microbial_assimilation": "hydrolysis",
            "hydrolysis_assisted_microbial_assimilation": "hydrolysis",
            "hydrolytic_degradation": "hydrolysis",
            "surface_erosiondriven_biodegradation": "surface",
            "surface_erosion_driven_biodegradation": "surface",
            "crackenhanced_microbial_biodegradation": "surface",
            "environmentally_modified_microbial_degradation": "other",
            "stabilization_or_degradation_inhibition_due_to_crosslinking": "other",
        },
        "degradation_environment": {
            "soil": "soil", "field_soil": "soil", "marine": "marine_aquatic",
            "freshwater": "marine_aquatic", "industrial_compost": "compost_organic_fertilizer",
            "compost": "compost_organic_fertilizer", "thermophilic_compost": "compost_organic_fertilizer",
            "vermicompost": "compost_organic_fertilizer", "organic_fertilizer": "compost_organic_fertilizer",
            "liquid_mineral_medium": "laboratory_mineral_media",
        },
        "sample_shape_morphology": {
            "flat_ribbons": "ribbons", "ribbons": "ribbons", "films": "films",
            "circular_blend_films": "films", "flat_plates_and_thin_films": "films",
            "thin_films": "films", "biocomposite_sheets": "sheets", "laminates": "sheets",
            "sheets": "sheets", "rectangular_plates": "sheets",
            "rectangular_strips_flat_sheets": "sheets", "filaments_pellets": "other",
            "films_spherulite_morphologies": "other", "spherulites": "other",
        },
        "pha_degrading_microbes": {
            "bacteria_and_fungi": "bacteria_dominated",
            "unspecified_marine_bacteria": "bacteria_dominated",
            "the_phas_degraders_were_identified_from_genus_cupriavidus_acidovorax_variovora_streptomyces_and_ideonella_and_two_starch_degraders_were_identified_from_genus_moraxella_and_alcaligenes": "bacteria_dominated",
            "mixed_mostly_bacteria": "bacteria_dominated", "p_mendocina_ds04t": "bacteria_dominated",
            "marine_bacteria": "bacteria_dominated", "thermophilic_bacteria_fungi": "bacteria_dominated",
            "fungi": "fungi_dominated", "fungal_mycelium": "fungi_dominated",
            "fungal_hyphae_and_spores": "fungi_dominated",
            "fungi_yeasts_and_bacteria": "bacteria_fungi_mixed",
            "bacteria_fungi_actinomycetes": "bacteria_fungi_mixed",
            "bacteria_actinomycetes_streptomyces_and_fungal_mycelium": "bacteria_fungi_mixed",
            "mixed_bacteria_fungi": "bacteria_fungi_mixed",
            "diverse_microbial_community": "diverse_environmental",
            "aquatic_microbes_bacteria_algae_etc": "diverse_environmental",
            "bacteria_and_possibly_fungi_or_algae": "diverse_environmental",
            "mesophilic_microbes_eisenia_foetida": "uncategorized",
        },
        "additive_type_1": {
            "plasticizer": "plasticizer", "filler": "filler", "reinforcement": "reinforcement",
            "crosslinker": "crosslinker_polymer_modifier",
            "blend_polymer": "crosslinker_polymer_modifier", "not_applicable": "not_applicable",
        },
    }
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "source_category": source,
                "modeled_category": modeled,
                "provenance": "origin_data/Weight-loss-model-main/weight_loss_data_preprocessing_LM.ipynb",
            }
            for feature, feature_mapping in mappings.items()
            for source, modeled in feature_mapping.items()
        ]
    )


def data_dictionary(table: pd.DataFrame) -> pd.DataFrame:
    """Create a compact, machine-readable dictionary for the processed table."""
    rows = []
    for column in table.columns:
        series = table[column]
        if column in TRACEABILITY_COLUMNS:
            role = "traceability_only"
        elif column in TARGET_COLUMNS:
            role = "target_or_target_audit"
        elif column in {
            "degradation_time_raw_days",
            "degradation_time_cleaned_days",
            "time_absolute_difference_days",
            "time_truncation_matches",
            "temperature_original_c",
            "t_deg_source_completed",
            "temperature_was_source_imputed",
        } or column.endswith("_source_completed") or column.endswith("_was_source_zero_filled"):
            role = "reconstruction_audit"
        elif column in MODEL_FEATURES:
            role = "candidate_model_feature"
        else:
            role = "reconstruction_audit"
        definition, unit = COLUMN_METADATA.get(
            column,
            (column.replace("_", " ").capitalize() + ".", "not applicable"),
        )
        derivation = DERIVATION_NOTES.get(
            column,
            (
                "Inherited directly from the released cleaned modelling table; no new derivation was applied."
                if column in MODEL_FEATURES
                else "Derived or normalised by the positional reconstruction and audit pipeline."
            ),
        )
        missingness_meaning = MISSINGNESS_NOTES.get(
            column,
            (
                "Missing denotes an unreported or unavailable source value; fold-fitted preprocessing handles it when this feature is modelled."
                if column in MODEL_FEATURES
                else "Missing denotes unavailable source traceability or an inapplicable derived audit value."
            ),
        )
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "role": role,
                "definition": definition,
                "unit": unit,
                "source_field": column if column in MODEL_FEATURES else "derived_or_traceability",
                "provenance": (
                    "Released source modeling table; inherited preprocessing applies."
                    if column in MODEL_FEATURES
                    else "Reconstructed or derived by the present audit pipeline."
                ),
                "derivation": derivation,
                "missingness_meaning": missingness_meaning,
                "model_inclusion": "included" if column in MODEL_FEATURES else "excluded",
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "unique_non_missing": int(series.nunique(dropna=True)),
                "example": "" if series.dropna().empty else str(series.dropna().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def sheet_audit(workbook: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize shapes, missingness, duplicates, and inferred dtypes by worksheet."""
    rows: list[dict[str, object]] = []
    for sheet, frame in workbook.items():
        rows.append(
            {
                "scope": "sheet",
                "sheet": sheet,
                "column": "",
                "rows": len(frame),
                "columns": frame.shape[1],
                "dtype": "",
                "missing_count": int(frame.isna().sum().sum()),
                "missing_fraction": float(frame.isna().mean().mean()),
                "unique_non_missing": "",
                "duplicate_rows": int(frame.duplicated().sum()),
            }
        )
        for column in frame.columns:
            series = frame[column]
            rows.append(
                {
                    "scope": "column",
                    "sheet": sheet,
                    "column": str(column),
                    "rows": len(frame),
                    "columns": frame.shape[1],
                    "dtype": str(series.dtype),
                    "missing_count": int(series.isna().sum()),
                    "missing_fraction": float(series.isna().mean()),
                    "unique_non_missing": int(series.nunique(dropna=True)),
                    "duplicate_rows": "",
                }
            )
    return pd.DataFrame(rows)


def curve_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize independent curves and observed local target decreases."""
    static_candidates = [
        "degradation_environment",
        "degradation_condition",
        "degradation_mechanism",
        "adjusted_hb_ratio_formulation_mol",
        "adjusted_hv_ratio_formulation_mol",
        "t_deg",
        "additives",
        "sample_shape_morphology",
        "experimental_scale",
    ]

    rows = []
    for curve_id, group in table.groupby("curve_id", sort=True):
        ordered = group.sort_values(["degradation_time_days", "curve_position"])
        deltas = ordered["weight_loss"].diff().dropna()
        decreases = deltas[deltas < 0]
        row: dict[str, object] = {
            "study_id": ordered["study_id_normalized"].iloc[0],
            "curve_id": curve_id,
            "instance": ordered["instance_normalized"].iloc[0],
            "sample_name": ordered["sample_name"].iloc[0],
            "n_time_points": len(ordered),
            "minimum_time_days": float(ordered["degradation_time_days"].min()),
            "maximum_time_days": float(ordered["degradation_time_days"].max()),
            "initial_weight_loss_pct": float(ordered["weight_loss"].iloc[0]),
            "final_weight_loss_pct": float(ordered["weight_loss"].iloc[-1]),
            "local_decrease_count": int((deltas < 0).sum()),
            "local_decrease_total_magnitude_pct": float(-decreases.sum()),
            "local_decrease_max_magnitude_pct": float(-decreases.min()) if len(decreases) else 0.0,
        }
        for column in static_candidates:
            if column in ordered:
                row[column] = ordered[column].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def hierarchy_metrics(table: pd.DataFrame, curves: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the core study/curve/time-point hierarchy metrics."""
    lengths = curves["n_time_points"]
    return {
        "studies": int(table["study_id_normalized"].nunique()),
        "curves": int(table["curve_id"].nunique()),
        "time_point_observations": int(len(table)),
        "curve_length_minimum": int(lengths.min()),
        "curve_length_median": float(lengths.median()),
        "curve_length_mean": float(lengths.mean()),
        "curve_length_maximum": int(lengths.max()),
        "time_minimum_days": float(table["degradation_time_days"].min()),
        "time_maximum_days": float(table["degradation_time_days"].max()),
        "target_minimum_pct": float(table["weight_loss"].min()),
        "target_maximum_pct": float(table["weight_loss"].max()),
        "curves_with_observed_decreases": int((curves["local_decrease_count"] > 0).sum()),
        "observed_local_decreases": int(curves["local_decrease_count"].sum()),
        "duplicated_feature_time_rows": int(
            table.drop(columns=list(TRACEABILITY_COLUMNS & set(table.columns)))
            .drop(columns=list(TARGET_COLUMNS & set(table.columns)))
            .duplicated(keep=False)
            .sum()
        ),
    }


def target_precision_metrics(table: pd.DataFrame) -> dict[str, float | int | str]:
    """Quantify precision lost by the source cleaned integer target."""
    difference = table["target_absolute_difference_pct"]
    small_raw = table["weight_loss_decimal"].between(0, 1, inclusive="neither")
    return {
        "changed_values": int((difference > 0).sum()),
        "changed_fraction": float((difference > 0).mean()),
        "mean_absolute_difference_pct": float(difference.mean()),
        "median_absolute_difference_pct": float(difference.median()),
        "maximum_absolute_difference_pct": float(difference.max()),
        "raw_values_between_0_and_1_pct": int(small_raw.sum()),
        "raw_values_between_0_and_1_converted_to_0": int(
            (small_raw & table["weight_loss_cleaned"].eq(0)).sum()
        ),
        "conversion_rule": "truncation_toward_zero",
        "exact_truncation_match_fraction": float(table["target_truncation_matches"].mean()),
    }
