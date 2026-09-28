"""Reconstruct raw decimal time/target pairs and join them to cleaned features."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from src.data_io import normalize_identifier


TIME_SHEET = "X_degradation_time_points"
TARGET_SHEET = "Y_weight_mass_loss_%"
ID_COLUMNS = 3


@dataclass(frozen=True)
class ReconstructionSummary:
    raw_pairs: int
    cleaned_rows: int
    matched_rows: int
    unmatched_rows: int
    excluded_raw_pairs: int
    time_truncation_matches: int
    target_truncation_matches: int


def _numeric(value: object) -> float | None:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(converted) else float(converted)


def reconstruct_raw_pairs(workbook: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert positionally aligned wide time/target worksheets to long format."""
    times = workbook[TIME_SHEET]
    targets = workbook[TARGET_SHEET]
    if times.shape != targets.shape:
        raise ValueError(f"Time and target worksheets differ in shape: {times.shape} vs {targets.shape}")

    records: list[dict[str, object]] = []
    for raw_row in range(len(times)):
        time_row = times.iloc[raw_row]
        target_row = targets.iloc[raw_row]
        time_key = tuple(normalize_identifier(time_row.iloc[i]) for i in (0, 1))
        target_key = tuple(normalize_identifier(target_row.iloc[i]) for i in (0, 1))
        if time_key != target_key:
            raise ValueError(f"Time/target identifier mismatch at source row {raw_row}: {time_key} vs {target_key}")

        sample_time = str(time_row.iloc[2]).strip()
        sample_target = str(target_row.iloc[2]).strip()
        sample_time_normalized = re.sub(r"[^a-z0-9]+", "", sample_time.lower())
        sample_target_normalized = re.sub(r"[^a-z0-9]+", "", sample_target.lower())
        if sample_time_normalized != sample_target_normalized:
            raise ValueError(
                f"Time/target sample-name mismatch at source row {raw_row}: "
                f"{sample_time!r} vs {sample_target!r}"
            )

        curve_position = 0
        for source_column_position in range(ID_COLUMNS, times.shape[1]):
            raw_time = _numeric(time_row.iloc[source_column_position])
            raw_target = _numeric(target_row.iloc[source_column_position])
            if (raw_time is None) != (raw_target is None):
                raise ValueError(
                    "Unpaired time/target cell at source row "
                    f"{raw_row}, column position {source_column_position}"
                )
            if raw_time is None:
                continue
            records.append(
                {
                    "study_id_normalized": time_key[0],
                    "instance_normalized": time_key[1],
                    "sample_name": sample_time,
                    "sample_name_target_sheet": sample_target,
                    "sample_name_exact_match": sample_time == sample_target,
                    "source_row": raw_row,
                    "source_column_position": source_column_position,
                    "curve_position": curve_position,
                    "degradation_time_raw_days": raw_time,
                    "weight_loss_decimal": raw_target,
                }
            )
            curve_position += 1

    pairs = pd.DataFrame.from_records(records)
    key = ["study_id_normalized", "instance_normalized", "curve_position"]
    if pairs.duplicated(key).any():
        raise ValueError("Raw curve-position keys are not unique")
    return pairs


def join_cleaned_features(
    cleaned: pd.DataFrame, raw_pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, ReconstructionSummary]:
    """Match cleaned rows to raw pairs and retain an explicit discrepancy table.

    The source preprocessing truncates both time and target to integers. Truncation
    creates duplicate time keys within some curves, so curve position is the only
    one-to-one key. Study, instance, source order, <1-day tolerance, and exact
    truncation equality are all verified before accepting a match.
    """
    required = {"study_id", "instance", "degradation_time_days", "weight_loss"}
    missing = sorted(required.difference(cleaned.columns))
    if missing:
        raise ValueError(f"Cleaned table is missing required columns: {missing}")

    table = cleaned.copy()
    table.insert(0, "source_cleaned_row", np.arange(len(table), dtype=int))
    table["study_id_normalized"] = table["study_id"].map(normalize_identifier)
    table["instance_normalized"] = table["instance"].map(normalize_identifier)
    table["curve_position"] = table.groupby(
        ["study_id_normalized", "instance_normalized"], sort=False
    ).cumcount()

    keys = ["study_id_normalized", "instance_normalized", "curve_position"]
    merged = table.merge(raw_pairs, on=keys, how="left", validate="one_to_one", indicator=True)
    merged["time_absolute_difference_days"] = (
        merged["degradation_time_raw_days"] - merged["degradation_time_days"]
    ).abs()
    merged["target_absolute_difference_pct"] = (
        merged["weight_loss_decimal"] - merged["weight_loss"]
    ).abs()
    merged["time_truncation_matches"] = (
        np.trunc(merged["degradation_time_raw_days"]) == merged["degradation_time_days"]
    )
    merged["target_truncation_matches"] = (
        np.trunc(merged["weight_loss_decimal"]) == merged["weight_loss"]
    )

    accepted = (
        merged["_merge"].eq("both")
        & merged["time_absolute_difference_days"].lt(1.0 + 1e-12)
        & merged["time_truncation_matches"]
        & merged["target_truncation_matches"]
    )
    discrepancies = merged.loc[~accepted].copy()
    if not discrepancies.empty:
        raise ValueError(
            f"Reconstruction validation failed for {len(discrepancies)} cleaned rows; "
            "inspect the discrepancy output"
        )

    matched_raw_keys = merged.loc[accepted, keys]
    raw_check = raw_pairs.merge(matched_raw_keys, on=keys, how="left", indicator=True)
    excluded_raw = raw_check.loc[raw_check["_merge"].eq("left_only")].drop(columns="_merge")
    excluded_raw.insert(0, "reason", "source curve excluded from cleaned modeling table")

    result = merged.loc[accepted].drop(columns="_merge").copy()
    result["curve_id"] = (
        "study_" + result["study_id_normalized"] + "__instance_" + result["instance_normalized"]
    )
    result["row_id"] = result["source_cleaned_row"].map(lambda value: f"row_{value:04d}")
    result = result.rename(
        columns={
            "degradation_time_days": "degradation_time_cleaned_days",
            "weight_loss": "weight_loss_cleaned",
        }
    )
    # Both time and target were truncated by the released preprocessing notebook.
    # The reconstructed decimal measurements are therefore the canonical analysis
    # variables; the released integer values remain explicit sensitivity fields.
    result["degradation_time_days"] = result["degradation_time_raw_days"]
    result["weight_loss"] = result["weight_loss_decimal"]

    ordered_front = [
        "row_id",
        "curve_id",
        "study_id_normalized",
        "instance_normalized",
        "sample_name",
        "curve_position",
        "source_cleaned_row",
        "source_row",
        "source_column_position",
        "degradation_time_raw_days",
        "degradation_time_days",
        "degradation_time_cleaned_days",
        "weight_loss_decimal",
        "weight_loss_cleaned",
        "weight_loss",
    ]
    remainder = [column for column in result.columns if column not in ordered_front]
    result = result[ordered_front + remainder]

    summary = ReconstructionSummary(
        raw_pairs=len(raw_pairs),
        cleaned_rows=len(cleaned),
        matched_rows=len(result),
        unmatched_rows=len(discrepancies),
        excluded_raw_pairs=len(excluded_raw),
        time_truncation_matches=int(result["time_truncation_matches"].sum()),
        target_truncation_matches=int(result["target_truncation_matches"].sum()),
    )
    return result, excluded_raw, summary


def restore_source_missingness(
    table: pd.DataFrame, workbook: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Restore temperature and additive-concentration reporting states.

    The released table globally completed missing temperatures and converted
    unreported additive concentrations to zero.  This function conserves those
    inherited values in traceability columns, reconstructs the workbook values,
    and exposes explicit reporting/ambiguity indicators for fold-local modeling.
    """
    result = table.copy()
    keys = ["study_id_normalized", "instance_normalized"]
    lookup_index = pd.MultiIndex.from_frame(result[keys])

    environment = workbook["Environmental Properties"].copy()
    environment = environment.loc[
        environment["Study_id"].notna() & environment["Instance"].notna()
    ]
    environment["study_id_normalized"] = environment["Study_id"].map(normalize_identifier)
    environment["instance_normalized"] = environment["Instance"].map(normalize_identifier)
    temperature_by_curve = environment.groupby(keys, sort=False)["T_deg"].first()

    result["t_deg_source_completed"] = result["t_deg"]
    result["temperature_original_c"] = lookup_index.map(temperature_by_curve)
    result["temperature_was_reported"] = result["temperature_original_c"].notna().astype(int)
    result["temperature_was_source_imputed"] = (
        result["temperature_original_c"].isna() & result["t_deg_source_completed"].notna()
    ).astype(int)
    result["t_deg"] = result["temperature_original_c"]

    physicochemical = workbook["Physicochemical properties"].copy()
    physicochemical = physicochemical.loc[
        physicochemical["Study_id"].notna() & physicochemical["Instance"].notna()
    ].drop_duplicates(subset=["Study_id", "Instance"], keep="first")
    physicochemical["study_id_normalized"] = physicochemical["Study_id"].map(
        normalize_identifier
    )
    physicochemical["instance_normalized"] = physicochemical["Instance"].map(
        normalize_identifier
    )
    physicochemical = physicochemical.set_index(keys)

    for slot in (1, 2, 3):
        modeled_column = f"additive{slot}_percentage_wt"
        raw_column = f"Additive{slot}_percentage (wt%)"
        type_column = f"Additive_type_{slot}"
        result[f"{modeled_column}_source_completed"] = result[modeled_column]

        raw_values = pd.Series(lookup_index.map(physicochemical[raw_column]), index=result.index)
        raw_types = pd.Series(lookup_index.map(physicochemical[type_column]), index=result.index)
        numeric = pd.to_numeric(raw_values, errors="coerce")
        type_present = raw_types.fillna("").astype(str).str.strip().ne("")
        ambiguous = type_present & (numeric.isna() | numeric.eq(0.0))
        reported = type_present & numeric.notna() & numeric.gt(0.0)

        # A genuinely unused additive slot is a structural zero.  A named
        # additive with a zero, missing, or non-wt% value is retained as missing.
        restored = numeric.where(type_present, 0.0).mask(ambiguous)
        result[modeled_column] = restored
        result[f"additive{slot}_concentration_reported"] = reported.astype(int)
        result[f"additive{slot}_concentration_ambiguous"] = ambiguous.astype(int)
        result[f"additive{slot}_was_source_zero_filled"] = (
            raw_values.isna()
            & result[f"{modeled_column}_source_completed"].eq(0.0)
        ).astype(int)

    return result
