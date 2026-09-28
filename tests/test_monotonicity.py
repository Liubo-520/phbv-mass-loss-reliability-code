import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models import BoundedTargetRegressor


def test_bounded_target_inverse_stays_in_physical_interval():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.linspace(-0.2, 100.4, 20)
    model = BoundedTargetRegressor(HistGradientBoostingRegressor(max_iter=20, random_state=42))
    model.fit(x, y)
    prediction = model.predict(np.array([[-1000.0], [1000.0]]))
    assert (prediction >= 0).all()
    assert (prediction <= 100).all()


def test_bounded_transform_clips_training_overshoot_and_inverse_extremes():
    model = BoundedTargetRegressor(HistGradientBoostingRegressor())
    transformed = model._forward(np.array([-2.0, 0.0, 100.0, 100.4]))
    assert transformed[0] == transformed[1]
    assert transformed[2] == transformed[3]
    prediction = model._inverse(np.array([-1e6, 0.0, 1e6]))
    assert np.all((prediction >= 0.0) & (prediction <= 100.0))


def test_monotonic_constraint_is_numerically_non_decreasing():
    from xgboost import XGBRegressor

    x = np.column_stack([np.linspace(0, 10, 60), np.tile([0.0, 1.0], 30)])
    y = 8 * x[:, 0] + 5 * np.sin(x[:, 0]) + x[:, 1]
    model = XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        monotone_constraints=(1, 0),
        tree_method="hist",
        random_state=42,
    )
    model.fit(x, y)
    grid = np.column_stack([np.linspace(0, 10, 200), np.zeros(200)])
    assert (np.diff(model.predict(grid)) >= -1e-9).all()
