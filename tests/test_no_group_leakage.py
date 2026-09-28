import pandas as pd

from src.data_io import ROOT
from src.splitting import (
    generate_curve_folds,
    generate_group_calibration_split,
    generate_loso_inner_folds,
    generate_random_row_split,
)


def _table():
    return pd.read_csv(ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv")


def test_random_row_split_has_no_row_overlap():
    split = generate_random_row_split(_table())
    assert split["row_id"].is_unique
    assert set(split["partition"]) == {"train", "test"}


def test_curve_grouped_folds_never_split_a_curve():
    split = generate_curve_folds(_table())
    assert split.groupby("curve_id")["test_fold"].nunique().max() == 1


def test_loso_inner_folds_never_split_a_study():
    inner = generate_loso_inner_folds(_table())
    counts = inner.groupby(["outer_test_study", "study_id"])["inner_validation_fold"].nunique()
    assert counts.max() == 1
    assert not (inner["outer_test_study"] == inner["study_id"]).any()


def test_calibration_curves_are_disjoint_from_fit_curves():
    assignment = generate_group_calibration_split(_table())
    counts = assignment.groupby(["outer_test_study", "curve_id"])["partition"].nunique()
    assert counts.max() == 1

