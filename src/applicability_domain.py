"""Training-only applicability-domain reliability indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor


def applicability_indicators(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Calculate kNN distance, range violations, and unseen categories from training only."""
    transformer = build_preprocessor(scale_numeric=True)
    train_matrix = transformer.fit_transform(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    test_matrix = transformer.transform(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    neighbors = NearestNeighbors(n_neighbors=min(5, len(train)), metric="euclidean")
    neighbors.fit(train_matrix)
    distances, _ = neighbors.kneighbors(test_matrix)

    numeric_minimum = train[NUMERIC_FEATURES].min()
    numeric_maximum = train[NUMERIC_FEATURES].max()
    range_violations = (
        test[NUMERIC_FEATURES].lt(numeric_minimum, axis="columns")
        | test[NUMERIC_FEATURES].gt(numeric_maximum, axis="columns")
    ).sum(axis=1)
    unseen = np.zeros(len(test), dtype=int)
    for column in CATEGORICAL_FEATURES:
        known = set(train[column].dropna().astype(str))
        unseen += ~test[column].astype(str).isin(known).to_numpy()
    return pd.DataFrame(
        {
            "row_id": test["row_id"].to_numpy(),
            "knn_distance_mean": distances.mean(axis=1),
            "knn_distance_max": distances.max(axis=1),
            "numeric_range_violation_count": range_violations.to_numpy(dtype=int),
            "unseen_category_count": unseen,
        }
    )

