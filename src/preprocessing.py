"""Leakage-safe feature definitions and preprocessing pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "degradation_time_days",
    "adjusted_hb_ratio_formulation_mol",
    "adjusted_hv_ratio_formulation_mol",
    "t_deg",
    "temperature_was_reported",
    "additive1_percentage_wt",
    "additive1_concentration_reported",
    "additive1_concentration_ambiguous",
    "additive2_percentage_wt",
    "additive2_concentration_reported",
    "additive2_concentration_ambiguous",
    "additive3_percentage_wt",
    "additive3_concentration_reported",
    "additive3_concentration_ambiguous",
]

CATEGORICAL_FEATURES = [
    "degradation_condition",
    "degradation_mechanism",
    "additives",
    "degradation_environment",
    "additive_type_1",
    "additive_type_2",
    "additive_type_3",
    "sample_shape_morphology",
    "pha_degrading_microbes",
    "experimental_scale",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FORBIDDEN_MODEL_COLUMNS = {
    "study_id",
    "study_id_normalized",
    "instance",
    "instance_normalized",
    "curve_id",
    "row_id",
    "sample_name",
    "source_cleaned_row",
    "source_row",
    "source_column_position",
    "degradation_time_raw_days",
    "degradation_time_cleaned_days",
    "weight_loss",
    "weight_loss_decimal",
    "weight_loss_cleaned",
}


def build_preprocessor(
    scale_numeric: bool = False,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    impute_numeric: bool = True,
) -> ColumnTransformer:
    """Build a fold-local mixed-type transformer with unknown-category handling."""
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_steps = []
    if impute_numeric:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, numeric_features), ("categorical", categorical, categorical_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
