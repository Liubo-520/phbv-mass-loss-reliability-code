"""Audit all source worksheets and the reconstructed data hierarchy."""

from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.audit import category_crosswalk, curve_summary, data_dictionary, hierarchy_metrics, sheet_audit, target_precision_metrics
from src.data_io import ROOT, load_cleaned_csv, load_raw_workbook
from src.reconstruct_decimal_target import (
    join_cleaned_features,
    reconstruct_raw_pairs,
    restore_source_missingness,
)


def main() -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    workbook = load_raw_workbook()
    cleaned = load_cleaned_csv()
    raw_pairs = reconstruct_raw_pairs(workbook)
    processed, excluded, reconstruction = join_cleaned_features(cleaned, raw_pairs)
    processed = restore_source_missingness(processed, workbook)

    audit = sheet_audit(workbook)
    csv_summary = pd.DataFrame(
        [
            {
                "scope": "sheet",
                "sheet": "cleaned_csv",
                "column": "",
                "rows": len(cleaned),
                "columns": cleaned.shape[1],
                "dtype": "",
                "missing_count": int(cleaned.isna().sum().sum()),
                "missing_fraction": float(cleaned.isna().mean().mean()),
                "unique_non_missing": "",
                "duplicate_rows": int(cleaned.duplicated().sum()),
            }
        ]
    )
    pd.concat([audit, csv_summary], ignore_index=True).to_csv(results / "data_audit.csv", index=False)
    data_dictionary(processed).to_csv(results / "data_dictionary.csv", index=False)
    category_crosswalk().to_csv(results / "category_crosswalk.csv", index=False)

    manifest_path = ROOT / "origin_data" / "SHA256SUMS.txt"
    hash_records = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        source_path = ROOT / relative_path.strip()
        observed = hashlib.sha256(source_path.read_bytes()).hexdigest()
        hash_records.append(
            {
                "path": relative_path.strip(),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matches": observed == expected,
                "bytes": source_path.stat().st_size,
            }
        )
    if not all(record["matches"] for record in hash_records):
        raise ValueError("At least one immutable source file failed its SHA-256 check.")
    (results / "source_file_hashes.json").write_text(
        json.dumps({"algorithm": "SHA-256", "all_match": True, "files": hash_records}, indent=2),
        encoding="utf-8",
    )

    curves = curve_summary(processed)
    curves.to_csv(results / "curve_level_summary.csv", index=False)
    excluded.to_csv(results / "excluded_raw_observations.csv", index=False)

    hierarchy = hierarchy_metrics(processed, curves)
    precision = target_precision_metrics(processed)
    lines = [
        "# Data audit report",
        "",
        "## Verified source inventory",
        "",
        f"- Raw workbook sheets: {', '.join(workbook)}.",
        f"- Immutable source-file hashes verified: {len(hash_records)} of {len(hash_records)} matched `origin_data/SHA256SUMS.txt`.",
        f"- Cleaned modeling table: {len(cleaned):,} rows and {cleaned.shape[1]} columns.",
        f"- Raw paired time-target observations: {reconstruction.raw_pairs:,}.",
        f"- Reliably reconstructed cleaned observations: {reconstruction.matched_rows:,} (100% of cleaned rows).",
        f"- Raw observations excluded by the source cleaned table: {reconstruction.excluded_raw_pairs:,}.",
        "",
        "## Hierarchical structure",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in hierarchy.items())
    lines.extend(["", "## Decimal precision audit", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in precision.items())
    lines.extend(
        [
            "",
            "The cleaned table truncates both degradation time and weight loss toward zero. "
            "Because this creates duplicate integer time keys within some curves, reconstruction "
            "uses study, instance, and original curve position, then requires exact truncation "
            "agreement and an absolute time difference below one day. Both reconstructed decimal "
            "quantities are the primary modelling variables; the released truncated time is retained "
            "only as a sensitivity analysis.",
            "",
            "Original temperature missingness and ambiguous additive concentrations are restored from "
            "the raw worksheets. Imputation is fitted within training folds, and source-completed "
            "columns are retained only for explicit inherited-preprocessing sensitivity analyses.",
            "",
            "Study ID, instance, curve ID, sample name, and source positions are traceability "
            "variables and are excluded from predictive features.",
        ]
    )
    (results / "data_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
