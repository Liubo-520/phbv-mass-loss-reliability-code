"""Curve-level endpoint integrity auditing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _source_evidence(study: str, instance: str) -> tuple[str, str, str]:
    if study == "19":
        return (
            "mass_loss",
            "high",
            "Ramsay_et_al_1993_figures_2_3_axes_weight_loss_pct",
        )
    if study == "9" and instance in {"4", "8"}:
        return (
            "uncertain_orientation",
            "low",
            "primary_abstract_confirms_biodegradation_context_but_full_axis_unavailable",
        )
    if study == "9":
        return (
            "mass_loss_context_only",
            "medium",
            "workbook_parameter_label_and_primary_abstract",
        )
    return (
        "mass_loss",
        "medium",
        "raw_workbook_parameter_evaluated_weight_loss",
    )


def audit_curves(table: pd.DataFrame) -> pd.DataFrame:
    """Compute endpoint, orientation, monotonicity, and time-integrity fields."""
    rows: list[dict[str, object]] = []
    for curve_id, group in table.groupby("curve_id", sort=True):
        ordered = group.sort_values(["degradation_time_days", "curve_position"])
        time = ordered["degradation_time_days"].to_numpy(dtype=float)
        target = ordered["weight_loss"].to_numpy(dtype=float)
        deltas = np.diff(target)
        time_deltas = np.diff(time)
        decreases = deltas[deltas < 0]
        slope = float(np.polyfit(time, target, 1)[0]) if np.unique(time).size > 1 else np.nan
        endpoint_change = float(target[-1] - target[0])
        maximum_decrease = float(-decreases.min()) if decreases.size else 0.0
        above_100 = bool(np.any(target > 100.0))
        severe = endpoint_change <= -10.0 and slope < 0
        if severe:
            integrity_class = "severe_orientation_conflict"
        elif endpoint_change < -2.0 or above_100:
            integrity_class = "uncertain_semantics"
        elif decreases.size:
            integrity_class = "minor_non_monotonic_noise"
        else:
            integrity_class = "endpoint_consistent"

        study = str(ordered["study_id_normalized"].iloc[0])
        instance = str(ordered["instance_normalized"].iloc[0])
        semantics, evidence_grade, evidence_basis = _source_evidence(study, instance)
        rows.append(
            {
                "study_id": study,
                "instance": instance,
                "curve_id": curve_id,
                "sample_name": ordered["sample_name"].iloc[0],
                "observations": len(ordered),
                "first_observation_pct": float(target[0]),
                "first_observation_time_days": float(time[0]),
                "contains_true_t0": bool(np.any(time == 0.0)),
                "adjacent_decrease_count": int(decreases.size),
                "maximum_adjacent_decrease_pct": maximum_decrease,
                "endpoint_minus_start_pct": endpoint_change,
                "overall_linear_slope_pct_per_day": slope,
                "exceeds_100_pct": above_100,
                "duplicate_time_count": int(np.sum(time_deltas == 0.0)),
                "maximum_absolute_jump_pct": float(np.max(np.abs(deltas))) if deltas.size else 0.0,
                "abnormal_jump_over_25_pct": bool(np.any(np.abs(deltas) > 25.0)),
                "endpoint_integrity_class": integrity_class,
                "endpoint_semantics_assessment": semantics,
                "semantics_evidence_grade": evidence_grade,
                "semantics_evidence_basis": evidence_basis,
            }
        )
    return pd.DataFrame(rows)


def study19_source_table(table: pd.DataFrame) -> pd.DataFrame:
    """Create source-verified study 19 curve context from Ramsay et al. (1993)."""
    source = table[table["study_id_normalized"].astype(str).eq("19")]
    thickness_um = {"1a": 100, "1b": 100, "2": 150, "3": 150, "4": 150, "5": 800, "6": 800, "7": 800}
    starch_fraction = {"1a": 0, "1b": 0, "2": 50, "3": 25, "4": 0, "5": 50, "6": 25, "7": 0}
    figure = {"1a": "Figure 2", "1b": "Figure 2", "2": "Figure 3a", "3": "Figure 3a", "4": "Figure 3a", "5": "Figure 3b", "6": "Figure 3b", "7": "Figure 3b"}
    rows = []
    for instance, group in source.groupby("instance_normalized", sort=False):
        ordered = group.sort_values("degradation_time_days")
        iid = str(instance)
        rows.append(
            {
                "study_id": 19,
                "instance": iid,
                "curve_id": ordered["curve_id"].iloc[0],
                "sample_name": ordered["sample_name"].iloc[0],
                "thickness_um": thickness_um[iid],
                "oxygen_condition": ordered["degradation_condition"].iloc[0],
                "starch_fraction_wt_pct": starch_fraction[iid],
                "specimen_geometry": "solvent_cast_film" if iid in {"1a", "1b"} else "ASTM_Instron_test_bar",
                "source_figure": figure[iid],
                "last_observed_time_days": float(ordered["degradation_time_days"].iloc[-1]),
                "last_observed_weight_loss_pct": float(ordered["weight_loss"].iloc[-1]),
                "endpoint_value_provenance": "raw_workbook_digitisation_of_source_figure",
                "source_endpoint_axis": "Weight Loss (%)",
                "context_verification": "primary_article_full_text_and_figure_captions",
                "primary_source": "Ramsay_et_al_1993_Applied_and_Environmental_Microbiology_59_1242_1246",
                "doi": "10.1128/aem.59.4.1242-1246.1993",
            }
        )
    return pd.DataFrame(rows)


def study9_source_table(table: pd.DataFrame) -> pd.DataFrame:
    """Record what the accessible Choi and Park source can and cannot establish."""
    source = table[table["study_id_normalized"].astype(str).eq("9")]
    rows = []
    for instance, group in source.groupby("instance_normalized", sort=False):
        ordered = group.sort_values("degradation_time_days")
        iid = str(instance)
        delta = float(ordered["weight_loss"].iloc[-1] - ordered["weight_loss"].iloc[0])
        rows.append(
            {
                "study_id": 9,
                "instance": iid,
                "curve_id": ordered["curve_id"].iloc[0],
                "sample_name": ordered["sample_name"].iloc[0],
                "first_time_days": float(ordered["degradation_time_days"].iloc[0]),
                "first_value_pct": float(ordered["weight_loss"].iloc[0]),
                "last_time_days": float(ordered["degradation_time_days"].iloc[-1]),
                "last_value_pct": float(ordered["weight_loss"].iloc[-1]),
                "endpoint_change_pct": delta,
                "source_title": "Thermal_and_Mechanical_Properties_of_PHBV_Plasticized_by_Biodegradable_Soybean_Oils",
                "doi": "10.1002/masy.200350707",
                "source_access": "publisher_abstract_and_bibliographic_record_only",
                "source_verified_context": "PHBV_soybean_oil_blends_and_accelerated_compost_degradation",
                "source_does_not_verify": "endpoint_axis_orientation_or_remaining_mass_semantics",
                "orientation_assessment": (
                    "severe_conflict_with_cumulative_mass_loss_label" if iid in {"4", "8"} else "consistent_with_mass_loss"
                ),
                "primary_analysis_decision": "retain_unchanged",
                "100_minus_y_decision": "not_run_without_primary_source_proof",
            }
        )
    return pd.DataFrame(rows)
