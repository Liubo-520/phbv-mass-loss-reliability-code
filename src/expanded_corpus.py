"""Recover raw-workbook curves omitted by the released complete-case table."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.reconstruct_decimal_target import normalize_identifier


DEGRADATION_MAPPING = {
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
}
ENVIRONMENT_MAPPING = {
    "soil": "soil",
    "field_soil": "soil",
    "marine": "marine_aquatic",
    "freshwater": "marine_aquatic",
    "industrial_compost": "compost_organic_fertilizer",
    "compost": "compost_organic_fertilizer",
    "thermophilic_compost": "compost_organic_fertilizer",
    "vermicompost": "compost_organic_fertilizer",
    "organic_fertilizer": "compost_organic_fertilizer",
    "liquid_mineral_medium": "laboratory_mineral_media",
}
MORPHOLOGY_MAPPING = {
    "flat_ribbons": "ribbons",
    "ribbons": "ribbons",
    "films": "films",
    "circular_blend_films": "films",
    "flat_plates_and_thin_films": "films",
    "thin_films": "films",
    "biocomposite_sheets": "sheets",
    "laminates": "sheets",
    "sheets": "sheets",
    "rectangular_plates": "sheets",
    "rectangular_strips_flat_sheets": "sheets",
    "filaments_pellets": "other",
    "films_spherulite_morphologies": "other",
    "spherulites": "other",
}
MICROBE_MAPPING = {
    "bacteria_and_fungi": "bacteria_dominated",
    "unspecified_marine_bacteria": "bacteria_dominated",
    "marine_bacteria": "bacteria_dominated",
    "mixed_mostly_bacteria": "bacteria_dominated",
    "p_mendocina_ds04t": "bacteria_dominated",
    "thermophilic_bacteria_fungi": "bacteria_dominated",
    "fungi": "fungi_dominated",
    "fungal_mycelium": "fungi_dominated",
    "fungal_hyphae_and_spores": "fungi_dominated",
    "fungi_yeasts_and_bacteria": "bacteria_fungi_mixed",
    "bacteria_fungi_actinomycetes": "bacteria_fungi_mixed",
    "bacteria_actinomycetes_streptomyces_and_fungal_mycelium": "bacteria_fungi_mixed",
    "mixed_bacteria_fungi": "bacteria_fungi_mixed",
    "diverse_microbial_community": "diverse_environmental",
    "aquatic_microbes_bacteria_algae_etc": "diverse_environmental",
    "bacteria_and_possibly_fungi_or_algae": "diverse_environmental",
}
ADDITIVE1_MAPPING = {
    "plasticizer": "plasticizer",
    "filler": "filler",
    "reinforcement": "reinforcement",
    "crosslinker": "crosslinker_polymer_modifier",
    "blend_polymer": "crosslinker_polymer_modifier",
    "not_applicable": "not_applicable",
}


def clean_cell(value: object) -> object:
    """Replicate the released notebook's string normalisation."""
    if pd.isna(value):
        return np.nan
    if not isinstance(value, str):
        return value
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[%()]", "", cleaned).replace("/", "_")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or np.nan


def _indexed_source(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["Study_id"].notna() & frame["Instance"].notna()].copy()
    result["study_id_normalized"] = result["Study_id"].map(normalize_identifier)
    result["instance_normalized"] = result["Instance"].map(normalize_identifier)
    return result.drop_duplicates(
        subset=["study_id_normalized", "instance_normalized"], keep="first"
    ).set_index(["study_id_normalized", "instance_normalized"])


def _category(value: object, mapping: dict[str, str], fallback: str) -> str:
    key = clean_cell(value)
    return mapping.get(str(key), fallback) if not pd.isna(key) else fallback


def recover_excluded_curves(
    excluded_raw: pd.DataFrame,
    workbook: dict[str, pd.DataFrame],
    primary_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover model fields for the 79 raw pairs excluded upstream.

    Study 8 composition is source-recovered from the reported 70 wt% PHBV
    formulation containing PHBV with 12 mol% HV. Study 14 additive values are
    reported in vol%, so they remain missing rather than being mislabeled wt%.
    """
    phys = _indexed_source(workbook["Physicochemical properties"])
    env = _indexed_source(workbook["Environmental Properties"])
    rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []

    for _, source in excluded_raw.iterrows():
        sid = normalize_identifier(source["study_id_normalized"])
        iid = normalize_identifier(source["instance_normalized"])
        key = (sid, iid)
        prow, erow = phys.loc[key], env.loc[key]

        hb = pd.to_numeric(prow["Adjusted_HB_ratio_formulation (mol%)"], errors="coerce")
        hv = pd.to_numeric(prow["Adjusted_HV_ratio_formulation (mol%) "], errors="coerce")
        composition_source = "raw_workbook_adjusted_ratios"
        if sid == "8" and iid == "1":
            hb, hv = 61.6, 8.4
            composition_source = "Imam_et_al_1999_70wtpct_PHBV_with_12molpct_HV"

        record: dict[str, object] = {column: np.nan for column in primary_columns}
        record.update(source.to_dict())
        curve_id = f"study_{sid}__instance_{iid}"
        raw_time = float(source["degradation_time_raw_days"])
        raw_target = float(source["weight_loss_decimal"])
        record.update(
            {
                "row_id": f"expanded_{sid}_{iid}_{int(source['curve_position']):03d}",
                "curve_id": curve_id,
                "study_id": float(sid),
                "instance": iid,
                "study_id_normalized": sid,
                "instance_normalized": iid,
                "source_cleaned_row": np.nan,
                "degradation_time_days": raw_time,
                "degradation_time_cleaned_days": float(np.trunc(raw_time)),
                "weight_loss": raw_target,
                "weight_loss_cleaned": float(np.trunc(raw_target)),
                "adjusted_hb_ratio_formulation_mol": float(hb),
                "adjusted_hv_ratio_formulation_mol": float(hv),
                "degradation_condition": clean_cell(erow["Degradation_condition"]),
                "degradation_mechanism": _category(
                    erow["Degradation_mechanism"], DEGRADATION_MAPPING, "uncategorized"
                ),
                "additives": clean_cell(prow["Additives"]),
                "t_deg": pd.to_numeric(erow["T_deg"], errors="coerce"),
                "temperature_original_c": pd.to_numeric(erow["T_deg"], errors="coerce"),
                "temperature_was_reported": int(pd.notna(erow["T_deg"])),
                "temperature_was_source_imputed": int(pd.isna(erow["T_deg"])),
                "t_deg_source_completed": (
                    26.5 if sid == "14" and pd.isna(erow["T_deg"]) else erow["T_deg"]
                ),
                "degradation_environment": _category(
                    erow["Degradation_Environment"], ENVIRONMENT_MAPPING, "other"
                ),
                "sample_shape_morphology": _category(
                    prow["Sample_shape/Morphology"], MORPHOLOGY_MAPPING, "uncategorized"
                ),
                "pha_degrading_microbes": _category(
                    erow["PHA_degrading_microbes"], MICROBE_MAPPING, "uncategorized"
                ),
                "experimental_scale": clean_cell(erow["Experimental_Scale"]),
                "time_absolute_difference_days": abs(raw_time - float(np.trunc(raw_time))),
                "target_absolute_difference_pct": abs(raw_target - float(np.trunc(raw_target))),
                "time_truncation_matches": True,
                "target_truncation_matches": True,
                "expanded_recovery_status": "recovered_from_raw_workbook",
                "composition_recovery_source": composition_source,
            }
        )

        for slot in (1, 2, 3):
            raw_type = prow[f"Additive_type_{slot}"]
            type_clean = clean_cell(raw_type)
            type_present = pd.notna(type_clean)
            modeled_type = (
                ADDITIVE1_MAPPING.get(str(type_clean), "uncategorized")
                if slot == 1
                else str(type_clean)
            ) if type_present else "not_applicable"
            raw_value = prow[f"Additive{slot}_percentage (wt%)"]
            numeric = pd.to_numeric(raw_value, errors="coerce")
            ambiguous = bool(type_present and (pd.isna(numeric) or float(numeric) == 0.0))
            reported = bool(type_present and pd.notna(numeric) and float(numeric) > 0.0)
            restored = np.nan if ambiguous else (float(numeric) if type_present else 0.0)
            modeled_column = f"additive{slot}_percentage_wt"
            record[f"additive_type_{slot}"] = modeled_type
            record[modeled_column] = restored
            record[f"{modeled_column}_source_completed"] = numeric
            record[f"additive{slot}_concentration_reported"] = int(reported)
            record[f"additive{slot}_concentration_ambiguous"] = int(ambiguous)
            record[f"additive{slot}_was_source_zero_filled"] = int(
                pd.isna(raw_value) and not type_present
            )

        rows.append(record)
        if int(source["curve_position"]) == 0:
            missing_fields = [
                name
                for name in [
                    "adjusted_hb_ratio_formulation_mol",
                    "adjusted_hv_ratio_formulation_mol",
                    "t_deg",
                    "additive1_percentage_wt",
                    "additive2_percentage_wt",
                    "additive3_percentage_wt",
                ]
                if pd.isna(record[name])
            ]
            same_curve = (
                excluded_raw["study_id_normalized"].map(normalize_identifier).eq(sid)
                & excluded_raw["instance_normalized"].map(normalize_identifier).eq(iid)
            )
            recovery_rows.append(
                {
                    "study_id": sid,
                    "instance": iid,
                    "curve_id": curve_id,
                    "sample_name": source["sample_name"],
                    "observations_recovered": int(same_curve.sum()),
                    "upstream_exclusion_reason": (
                        "missing_adjusted_composition"
                        if sid == "8"
                        else "volpct_additive_concentration_coerced_to_missing"
                    ),
                    "composition_recovery": composition_source,
                    "composition_source_doi": (
                        "10.1128/AEM.65.2.431-437.1999" if sid == "8" else "10.3389/fmats.2019.00275"
                    ),
                    "remaining_missing_model_fields": ";".join(missing_fields),
                    "concentration_unit_decision": (
                        "not_converted_volpct_to_wtpct" if sid == "14" else "reported_wtpct"
                    ),
                }
            )

    recovered = pd.DataFrame(rows)
    ordered = primary_columns + [column for column in recovered if column not in primary_columns]
    return recovered.loc[:, ordered], pd.DataFrame(recovery_rows)


def build_expanded_corpus(
    primary: pd.DataFrame,
    excluded_raw: pd.DataFrame,
    workbook: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the primary complete-case corpus with recovered raw curves."""
    recovered, recovery = recover_excluded_curves(excluded_raw, workbook, primary.columns.tolist())
    expanded = pd.concat([primary, recovered], ignore_index=True, sort=False)
    expanded = expanded.sort_values(
        ["study_id_normalized", "instance_normalized", "curve_position"],
        key=lambda series: series.astype(str),
    ).reset_index(drop=True)
    hierarchy = (
        len(expanded),
        expanded["study_id_normalized"].nunique(),
        expanded["curve_id"].nunique(),
    )
    if hierarchy != (1546, 17, 133):
        raise ValueError(f"Unexpected expanded hierarchy: {hierarchy!r}.")
    return expanded, recovery
