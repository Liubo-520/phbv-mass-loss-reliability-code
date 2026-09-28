"""Run fixed-hyperparameter baselines on identical leakage-aware folds."""

from __future__ import annotations

import io
import json
import pickle
import sys
import time
import tracemalloc
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_io import ROOT
from src.evaluation import summarize_prediction_table
from src.models import backend_status, build_pipeline, screening_model_specs
from src.preprocessing import MODEL_FEATURES


def _folds(table: pd.DataFrame):
    random = pd.read_csv(ROOT / "splits" / "task_a_random_row.csv")
    train_ids = set(random.loc[random.partition.eq("train"), "row_id"])
    test_ids = set(random.loc[random.partition.eq("test"), "row_id"])
    yield "random_row", "seed_42", table.row_id.isin(train_ids), table.row_id.isin(test_ids)

    curve = pd.read_csv(ROOT / "splits" / "task_b_curve_grouped.csv")
    curve_fold = table[["row_id"]].merge(curve[["row_id", "test_fold"]], on="row_id", validate="one_to_one")
    for fold_id in sorted(curve_fold.test_fold.unique()):
        test = curve_fold.test_fold.eq(fold_id).to_numpy()
        yield "curve_grouped", f"fold_{fold_id}", ~test, test

    for study in sorted(table.study_id_normalized.astype(str).unique()):
        test = table.study_id_normalized.astype(str).eq(study).to_numpy()
        yield "leave_one_study_out", f"study_{study}", ~test, test


def _serialized_size_bytes(model) -> int:
    buffer = io.BytesIO()
    pickle.dump(model, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    return buffer.tell()


def main() -> None:
    table = pd.read_csv(
        ROOT / "data" / "processed" / "phbv_weight_loss_decimal.csv",
        dtype={"study_id_normalized": str, "instance_normalized": str},
    )
    prediction_rows = []
    runtime_rows = []
    specs = screening_model_specs()

    for spec in specs:
        for task, fold, train_mask, test_mask in _folds(table):
            train = table.loc[train_mask]
            test = table.loc[test_mask]
            pipeline = build_pipeline(spec)

            tracemalloc.start()
            fit_start = time.perf_counter()
            pipeline.fit(train[MODEL_FEATURES], train["weight_loss"])
            fit_seconds = time.perf_counter() - fit_start
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            predict_start = time.perf_counter()
            prediction = pipeline.predict(test[MODEL_FEATURES])
            predict_seconds = time.perf_counter() - predict_start
            runtime_rows.append(
                {
                    "task": task,
                    "fold": fold,
                    "model": spec.name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "python_peak_memory_bytes": peak_bytes,
                    "serialized_model_bytes": _serialized_size_bytes(pipeline),
                }
            )
            for (_, row), y_pred in zip(test.iterrows(), prediction):
                prediction_rows.append(
                    {
                        "task": task,
                        "fold": fold,
                        "model": spec.name,
                        "row_id": row.row_id,
                        "study_id": str(row.study_id_normalized),
                        "curve_id": row.curve_id,
                        "curve_position": int(row.curve_position),
                        "degradation_time_days": float(row.degradation_time_days),
                        "degradation_environment": row.degradation_environment,
                        "y_true": float(row.weight_loss),
                        "y_pred": float(y_pred),
                    }
                )
            print(f"completed {spec.name}: {task}/{fold}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    metrics, curve_metrics, study_metrics = summarize_prediction_table(predictions)
    prediction_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"
    log_dir = ROOT / "results" / "logs"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_dir / "screening_predictions.csv", index=False)
    metrics.to_csv(metric_dir / "screening_metrics.csv", index=False)
    curve_metrics.to_csv(metric_dir / "screening_curve_metrics.csv", index=False)
    study_metrics.to_csv(metric_dir / "screening_study_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(metric_dir / "screening_runtime.csv", index=False)
    (log_dir / "model_availability.json").write_text(
        json.dumps(backend_status(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

