import numpy as np
import pandas as pd

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_MODEL_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def test_preprocessor_fits_numeric_imputation_on_training_rows_only():
    train = pd.DataFrame({column: [1.0, np.nan, 3.0] for column in NUMERIC_FEATURES})
    test = pd.DataFrame({column: [9999.0] for column in NUMERIC_FEATURES})
    for column in CATEGORICAL_FEATURES:
        train[column] = ["known", None, "known"]
        test[column] = ["unseen"]

    transformer = build_preprocessor(scale_numeric=True)
    transformer.fit(train[MODEL_FEATURES])
    numeric_imputer = transformer.named_transformers_["numeric"].named_steps["imputer"]
    assert np.allclose(numeric_imputer.statistics_, 2.0)

    transformed_test = transformer.transform(test[MODEL_FEATURES])
    assert transformed_test.shape[0] == 1
    assert np.isfinite(transformed_test).all()


def test_traceability_and_target_columns_are_not_model_features():
    assert not FORBIDDEN_MODEL_COLUMNS.intersection(MODEL_FEATURES)
    assert not any("doi" in column or "paper_title" in column for column in MODEL_FEATURES)
