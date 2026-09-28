"""Fixed leakage-aware split generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


def generate_random_row_split(table: pd.DataFrame, seed: int = 42, test_size: float = 0.2) -> pd.DataFrame:
    """Create the optimistic row-wise reproduction split."""
    train, test = train_test_split(table["row_id"], test_size=test_size, random_state=seed, shuffle=True)
    result = pd.DataFrame({"row_id": table["row_id"]})
    result["partition"] = "train"
    result.loc[result["row_id"].isin(test), "partition"] = "test"
    result["seed"] = seed
    return result


def generate_curve_folds(table: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Assign every complete curve to exactly one held-out fold."""
    groups = table["curve_id"]
    splitter = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    fold = pd.Series(index=table.index, dtype="int64")
    for fold_id, (_, test_idx) in enumerate(splitter.split(table, groups=groups)):
        fold.iloc[test_idx] = fold_id
    return pd.DataFrame({"row_id": table["row_id"], "curve_id": groups, "test_fold": fold.astype(int)})


def generate_loso_folds(table: pd.DataFrame) -> pd.DataFrame:
    """Assign every row to the outer fold named by its held-out study."""
    return pd.DataFrame(
        {
            "row_id": table["row_id"],
            "study_id": table["study_id_normalized"],
            "outer_test_study": table["study_id_normalized"],
        }
    )


def generate_loso_inner_folds(table: pd.DataFrame, n_splits: int = 3) -> pd.DataFrame:
    """Precompute study-grouped inner folds for each outer LOSO training set."""
    rows: list[pd.DataFrame] = []
    for outer_study in sorted(table["study_id_normalized"].unique(), key=str):
        outer_train = table.loc[table["study_id_normalized"].ne(outer_study)].copy()
        groups = outer_train["study_id_normalized"]
        splitter = GroupKFold(n_splits=min(n_splits, groups.nunique()))
        inner = pd.Series(index=outer_train.index, dtype="int64")
        for fold_id, (_, validation_idx) in enumerate(splitter.split(outer_train, groups=groups)):
            inner.iloc[validation_idx] = fold_id
        rows.append(
            pd.DataFrame(
                {
                    "outer_test_study": outer_study,
                    "row_id": outer_train["row_id"],
                    "study_id": groups,
                    "inner_validation_fold": inner.astype(int),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def generate_group_calibration_split(
    table: pd.DataFrame, seed: int = 42, calibration_fraction: float = 0.25
) -> pd.DataFrame:
    """Precompute curve-separated fit/calibration sets inside every outer LOSO fold."""
    outputs = []
    for outer_study in sorted(table["study_id_normalized"].unique(), key=str):
        outer_train = table.loc[table["study_id_normalized"].ne(outer_study)].copy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=calibration_fraction, random_state=seed)
        fit_idx, calibration_idx = next(
            splitter.split(outer_train, groups=outer_train["curve_id"])
        )
        assignment = pd.DataFrame(
            {
                "outer_test_study": outer_study,
                "row_id": outer_train["row_id"],
                "curve_id": outer_train["curve_id"],
                "partition": "fit",
            }
        )
        assignment.iloc[calibration_idx, assignment.columns.get_loc("partition")] = "calibration"
        outputs.append(assignment)
    return pd.concat(outputs, ignore_index=True)


def save_split_metadata(path: Path, metadata: dict[str, object]) -> None:
    """Save split configuration in deterministic JSON form."""
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

