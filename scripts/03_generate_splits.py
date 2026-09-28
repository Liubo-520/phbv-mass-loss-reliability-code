"""Generate all fixed split assignments before model training."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_io import ROOT
from src.splitting import (
    generate_curve_folds,
    generate_group_calibration_split,
    generate_loso_folds,
    generate_loso_inner_folds,
    generate_random_row_split,
    save_split_metadata,
)


def main() -> None:
    table = pd.read_csv(ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv")
    split_dir = ROOT / "splits"
    split_dir.mkdir(exist_ok=True)

    random_split = generate_random_row_split(table)
    curve_folds = generate_curve_folds(table)
    loso = generate_loso_folds(table)
    inner = generate_loso_inner_folds(table)
    calibration = generate_group_calibration_split(table)

    random_split.to_csv(split_dir / "task_a_random_row.csv", index=False)
    curve_folds.to_csv(split_dir / "task_b_curve_grouped.csv", index=False)
    loso.to_csv(split_dir / "task_c_loso.csv", index=False)
    inner.to_csv(split_dir / "task_c_inner_grouped.csv", index=False)
    calibration.to_csv(split_dir / "task_c_calibration_groups.csv", index=False)
    save_split_metadata(
        split_dir / "split_metadata.json",
        {
            "seed": 42,
            "task_a": {"strategy": "random row", "test_fraction": 0.2},
            "task_b": {"strategy": "GroupKFold by curve_id", "folds": int(curve_folds.test_fold.nunique())},
            "task_c": {"strategy": "leave one complete study out", "folds": int(loso.outer_test_study.nunique())},
            "task_c_inner": {"strategy": "GroupKFold by study within outer training data", "folds": 3},
            "calibration": {"strategy": "GroupShuffleSplit by curve within outer training data", "fraction": 0.25},
        },
    )


if __name__ == "__main__":
    main()
