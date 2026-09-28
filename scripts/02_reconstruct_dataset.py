"""Build the canonical decimal-precision modeling table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.audit import target_precision_metrics
from src.data_io import ROOT, load_cleaned_csv, load_raw_workbook
from src.expanded_corpus import build_expanded_corpus
from src.reconstruct_decimal_target import (
    join_cleaned_features,
    reconstruct_raw_pairs,
    restore_source_missingness,
)


def main() -> None:
    processed_dir = ROOT / "data" / "processed"
    results_dir = ROOT / "results"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    workbook = load_raw_workbook()
    raw_pairs = reconstruct_raw_pairs(workbook)
    table, excluded, summary = join_cleaned_features(load_cleaned_csv(), raw_pairs)
    table = restore_source_missingness(table, workbook)
    expanded, recovery = build_expanded_corpus(table, excluded, workbook)
    table.to_csv(processed_dir / "phbv_weight_loss_decimal.csv", index=False)
    expanded.to_csv(processed_dir / "phbv_weight_loss_expanded.csv", index=False)
    excluded.to_csv(results_dir / "reconstruction_exclusions.csv", index=False)
    recovery.to_csv(results_dir / "expanded_corpus_recovery.csv", index=False)
    pd.DataFrame(
        {
            "column": expanded.columns,
            "missing": [int(expanded[column].isna().sum()) for column in expanded],
            "missing_fraction": [float(expanded[column].isna().mean()) for column in expanded],
        }
    ).to_csv(results_dir / "expanded_corpus_feature_missingness.csv", index=False)

    comparison_columns = [
        "row_id",
        "study_id_normalized",
        "curve_id",
        "degradation_environment",
        "degradation_time_raw_days",
        "degradation_time_days",
        "degradation_time_cleaned_days",
        "weight_loss_decimal",
        "weight_loss_cleaned",
        "time_absolute_difference_days",
        "target_absolute_difference_pct",
    ]
    table[comparison_columns].to_csv(results_dir / "raw_vs_cleaned_target_comparison.csv", index=False)

    metrics = target_precision_metrics(table)
    payload = {"reconstruction": summary.__dict__, "precision": metrics}
    (results_dir / "reconstruction_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Decimal target reconstruction report",
        "",
        f"All {summary.cleaned_rows:,} cleaned rows were mapped one-to-one to source workbook pairs.",
        f"The raw workbook contains {summary.raw_pairs:,} paired observations; "
        f"{summary.excluded_raw_pairs:,} belong to curves excluded from the cleaned modeling table.",
        "",
        "The source cleaned CSV used truncation toward zero for both time and target. The primary "
        "analysis therefore uses reconstructed raw-decimal `degradation_time_days` and "
        "`weight_loss_decimal`; `degradation_time_cleaned_days` and `weight_loss_cleaned` are "
        "retained only for sensitivity analysis.",
        "",
        f"Original temperature reporting was restored for "
        f"{int(table['temperature_was_reported'].sum()):,} rows; "
        f"{int(table['temperature_was_source_imputed'].sum()):,} rows had been globally completed "
        "by the released workflow and are now imputed only within training folds.",
        "",
        f"The expanded-corpus sensitivity recovers {len(excluded):,} observations and reaches "
        f"{len(expanded):,} observations, {expanded['curve_id'].nunique()} curves, and "
        f"{expanded['study_id_normalized'].nunique()} studies. Study 14 vol% additive values remain "
        "missing rather than being relabeled as wt%.",
        "",
    ]
    report.extend(f"- `{key}`: {value}" for key, value in metrics.items())
    (results_dir / "reconstruction_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
