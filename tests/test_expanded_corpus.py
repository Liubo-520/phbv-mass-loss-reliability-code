from src.data_io import load_cleaned_csv, load_raw_workbook
from src.expanded_corpus import build_expanded_corpus
from src.reconstruct_decimal_target import (
    join_cleaned_features,
    reconstruct_raw_pairs,
    restore_source_missingness,
)


def test_expanded_corpus_recovers_expected_hierarchy_and_missingness():
    workbook = load_raw_workbook()
    primary, excluded, _ = join_cleaned_features(
        load_cleaned_csv(), reconstruct_raw_pairs(workbook)
    )
    primary = restore_source_missingness(primary, workbook)
    expanded, recovery = build_expanded_corpus(primary, excluded, workbook)

    assert (len(expanded), expanded["curve_id"].nunique()) == (1546, 133)
    assert expanded["study_id_normalized"].nunique() == 17
    study8 = expanded[expanded["study_id_normalized"].astype(str).eq("8")]
    assert set(study8["adjusted_hb_ratio_formulation_mol"]) == {61.6}
    assert set(study8["adjusted_hv_ratio_formulation_mol"]) == {8.4}
    study14_recovered = expanded[
        expanded["curve_id"].isin(
            ["study_14__instance_2", "study_14__instance_3", "study_14__instance_4"]
        )
    ]
    assert study14_recovered["additive1_percentage_wt"].isna().all()
    assert set(recovery["observations_recovered"]) == {6, 18, 26, 29}
