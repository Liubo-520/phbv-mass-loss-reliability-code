"""Traditional regression models and scientifically motivated wrappers."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.preprocessing import build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], BaseEstimator]
    scale_numeric: bool = False


def _xgboost_factory() -> BaseEstimator:
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def _lightgbm_factory() -> BaseEstimator:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )


def _catboost_factory() -> BaseEstimator:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=400,
        depth=7,
        learning_rate=0.04,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def backend_status() -> dict[str, str]:
    """Return import status for optional boosted-tree libraries."""
    result = {}
    for module in ("xgboost", "lightgbm", "catboost"):
        try:
            imported = importlib.import_module(module)
            result[module] = f"available:{getattr(imported, '__version__', 'unknown')}"
        except Exception as exc:  # pragma: no cover - depends on local environment
            result[module] = f"unavailable:{type(exc).__name__}:{exc}"
    return result


def screening_model_specs() -> list[ModelSpec]:
    """Return fixed, fast baseline models, skipping unavailable optional backends."""
    specs = [
        ModelSpec("global_mean", lambda: DummyRegressor(strategy="mean")),
        ModelSpec("linear_regression", LinearRegression, scale_numeric=True),
        ModelSpec("ridge", lambda: Ridge(alpha=10.0), scale_numeric=True),
        ModelSpec(
            "elastic_net",
            lambda: ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=20_000, random_state=42),
            scale_numeric=True,
        ),
        ModelSpec("rbf_svr", lambda: SVR(C=30.0, epsilon=0.1, gamma="scale"), scale_numeric=True),
        ModelSpec(
            "random_forest",
            lambda: RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        ModelSpec(
            "extra_trees",
            lambda: ExtraTreesRegressor(
                n_estimators=250,
                min_samples_leaf=1,
                max_features=0.9,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        ModelSpec(
            "hist_gradient_boosting",
            lambda: HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ]
    optional = [
        ("xgboost", "xgboost", _xgboost_factory),
        ("lightgbm", "lightgbm", _lightgbm_factory),
        ("catboost", "catboost", _catboost_factory),
    ]
    status = backend_status()
    for name, module, factory in optional:
        if status[module].startswith("available"):
            specs.append(ModelSpec(name, factory))
    return specs


def build_pipeline(
    spec: ModelSpec,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    impute_numeric: bool = True,
) -> Pipeline:
    """Create a complete fold-local preprocessing and regression pipeline."""
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    spec.scale_numeric,
                    numeric_features,
                    categorical_features,
                    impute_numeric=impute_numeric,
                ),
            ),
            ("model", spec.factory()),
        ]
    )


def tuned_model_spec(name: str, params: dict[str, object], seed: int = 42) -> ModelSpec:
    """Build one of the three eligible tuned boosting estimators."""
    if name == "xgboost":
        def factory():
            from xgboost import XGBRegressor

            return XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                random_state=seed,
                **params,
            )

    elif name == "catboost":
        def factory():
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                loss_function="RMSE",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
                bootstrap_type="Bernoulli",
                **params,
            )

    elif name == "hist_gradient_boosting":
        def factory():
            return HistGradientBoostingRegressor(random_state=seed, **params)

    else:
        raise ValueError(f"Unsupported tuned model: {name}")
    return ModelSpec(name=name, factory=factory)


def supports_sample_weight(estimator: BaseEstimator) -> bool:
    """Check whether the final estimator exposes a sample_weight fit parameter."""
    try:
        signature = inspect.signature(estimator.fit)
    except (TypeError, ValueError):
        return False
    return "sample_weight" in signature.parameters


def sample_weights(table: pd.DataFrame, strategy: str) -> np.ndarray:
    """Compute fold-local weights for the requested study/curve estimand.

    ``study`` is retained as a backwards-compatible alias for the original
    two-level study--curve scheme.  The capped variant applies a fold-local
    95th-percentile cap after initial normalization and then renormalizes the
    result to mean one.
    """
    if strategy == "row":
        weights = np.ones(len(table), dtype=float)
    elif strategy == "curve":
        sizes = table.groupby("curve_id")["row_id"].transform("size").to_numpy(float)
        weights = 1.0 / sizes
    elif strategy == "study_only":
        study_sizes = (
            table.groupby("study_id_normalized")["row_id"].transform("size").to_numpy(float)
        )
        weights = 1.0 / study_sizes
    elif strategy in {"study", "study_curve", "study_curve_capped95"}:
        curve_sizes = table.groupby("curve_id")["row_id"].transform("size").to_numpy(float)
        curves_per_study = (
            table[["study_id_normalized", "curve_id"]]
            .drop_duplicates()
            .groupby("study_id_normalized")["curve_id"]
            .size()
        )
        n_curves = table["study_id_normalized"].map(curves_per_study).to_numpy(float)
        weights = 1.0 / (n_curves * curve_sizes)
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")
    weights = weights / weights.mean()
    if strategy == "study_curve_capped95":
        cap = float(np.quantile(weights, 0.95))
        weights = np.minimum(weights, cap)
        weights = weights / weights.mean()
    return weights


class BoundedTargetRegressor(BaseEstimator, RegressorMixin):
    """Fit on a logit-transformed, physically bounded target and invert predictions."""

    def __init__(self, estimator: BaseEstimator, epsilon: float = 1e-3):
        self.estimator = estimator
        self.epsilon = epsilon

    def _forward(self, y: np.ndarray) -> np.ndarray:
        bounded = np.clip(np.asarray(y, dtype=float), 0.0, 100.0)
        p = (bounded + self.epsilon) / (100.0 + 2.0 * self.epsilon)
        return np.log(p / (1.0 - p))

    def _inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        prediction = p * (100.0 + 2.0 * self.epsilon) - self.epsilon
        return np.clip(prediction, 0.0, 100.0)

    def fit(self, X, y, sample_weight=None):
        self.estimator_ = clone(self.estimator)
        transformed = self._forward(np.asarray(y))
        if sample_weight is not None and supports_sample_weight(self.estimator_):
            self.estimator_.fit(X, transformed, sample_weight=sample_weight)
        else:
            self.estimator_.fit(X, transformed)
        return self

    def predict(self, X):
        return self._inverse(self.estimator_.predict(X))


class MonotonicXGBRegressor(BaseEstimator, RegressorMixin):
    """XGBoost wrapper constraining only the first transformed feature (time)."""

    def __init__(self, params: dict[str, object] | None = None):
        self.params = params

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBRegressor

        constraints = tuple([1] + [0] * (X.shape[1] - 1))
        self.estimator_ = XGBRegressor(monotone_constraints=constraints, **(self.params or {}))
        self.estimator_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def xgboost_variant_spec(
    params: dict[str, object],
    *,
    bounded: bool,
    monotonic: bool,
    epsilon: float = 1e-3,
    seed: int = 42,
    name: str = "xgboost_variant",
) -> ModelSpec:
    """Build direct/bounded and unconstrained/monotonic XGBoost variants."""
    base_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
        **params,
    }

    def factory():
        if monotonic:
            estimator: BaseEstimator = MonotonicXGBRegressor(params=base_params)
        else:
            from xgboost import XGBRegressor

            estimator = XGBRegressor(**base_params)
        return BoundedTargetRegressor(estimator, epsilon=epsilon) if bounded else estimator

    return ModelSpec(name=name, factory=factory)
