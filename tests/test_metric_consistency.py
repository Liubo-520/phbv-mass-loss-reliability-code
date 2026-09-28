import numpy as np
import pandas as pd

from src.evaluation import point_metrics, summarize_prediction_table


def test_saved_prediction_metrics_are_reproducible():
    frame = pd.DataFrame(
        {
            "task": ["test"] * 4,
            "model": ["model"] * 4,
            "study_id": ["a", "a", "b", "b"],
            "curve_id": ["a1", "a1", "b1", "b1"],
            "curve_position": [0, 1, 0, 1],
            "degradation_time_days": [0, 1, 0, 1],
            "y_true": [0.0, 10.0, 20.0, 30.0],
            "y_pred": [1.0, 9.0, 18.0, 33.0],
        }
    )
    expected = point_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
    summary, _, _ = summarize_prediction_table(frame)
    assert np.isclose(summary.loc[0, "point_mae"], expected["mae"])
    assert np.isclose(summary.loc[0, "point_rmse"], expected["rmse"])
    assert np.isclose(summary.loc[0, "point_r2"], expected["r2"])

