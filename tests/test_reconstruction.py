import numpy as np

from src.data_io import load_cleaned_csv, load_raw_workbook
from src.reconstruct_decimal_target import join_cleaned_features, reconstruct_raw_pairs


def test_decimal_reconstruction_is_complete_and_one_to_one():
    cleaned = load_cleaned_csv()
    raw = reconstruct_raw_pairs(load_raw_workbook())
    table, excluded, summary = join_cleaned_features(cleaned, raw)

    assert summary.cleaned_rows == 1467
    assert summary.matched_rows == summary.cleaned_rows
    assert summary.unmatched_rows == 0
    assert table["row_id"].is_unique
    assert len(table) + len(excluded) == len(raw)


def test_source_cleaning_was_truncation_not_rounding():
    raw = reconstruct_raw_pairs(load_raw_workbook())
    table, _, _ = join_cleaned_features(load_cleaned_csv(), raw)

    assert np.array_equal(
        np.trunc(table["degradation_time_raw_days"]), table["degradation_time_cleaned_days"]
    )
    assert np.array_equal(table["degradation_time_raw_days"], table["degradation_time_days"])
    assert np.array_equal(np.trunc(table["weight_loss_decimal"]), table["weight_loss_cleaned"])
    assert (table["target_absolute_difference_pct"] < 1.0 + 1e-12).all()
    assert not np.array_equal(np.round(table["weight_loss_decimal"]), table["weight_loss_cleaned"])
