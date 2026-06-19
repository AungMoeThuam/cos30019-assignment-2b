from __future__ import annotations

import importlib.util
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("project root = " , PROJECT_ROOT)
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_data.csv"
EDGES_PATH = PROJECT_ROOT / "data" / "processed" / "edges.csv"
MODELS_DIR = PROJECT_ROOT / "models_saved"

PREDICTION_COLUMN = "predicted_hourly_traffic_volume"
EXPECTED_PREDICTION_COLUMNS = {
    "from_site",
    "to_site",
    "movement_id",
    "prediction_time_hhmm",
    "dayofweek",
    PREDICTION_COLUMN,
}

METRIC_THRESHOLDS = {
    "MAE": 55.0,
    "RMSE": 80.0,
    "R2": 0.70,
}

MODEL_CONFIG = {
    "random_forest": {
        "metrics": [
            MODELS_DIR / "random_forest_metrics.csv",
        ],
        "model": [
            MODELS_DIR / "random_forest_model.pkl",
        ],
        "metadata": [],
    },
    "lstm": {
        "metrics": [MODELS_DIR / "lstm_metrics_v4.csv"],
        "model": [MODELS_DIR / "lstm_model_v4.keras"],
        "metadata": [MODELS_DIR / "lstm_metadata_v4.joblib"],
    },
    "gru": {
        "metrics": [MODELS_DIR / "gru_metrics_v4.csv"],
        "model": [MODELS_DIR / "gru_model_v4.keras"],
        "metadata": [MODELS_DIR / "gru_metadata_v4.joblib"],
    },
}


def _first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    expected = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in paths)
    pytest.fail(f"Missing {label}. Expected one of: {expected}")


def _parse_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    pattern = re.compile(r"^(MAE|RMSE|R2)\s*:\s*(-?\d+(?:\.\d+)?)\s*$")

    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))

    missing = sorted(set(METRIC_THRESHOLDS) - set(metrics))
    assert not missing, f"{path} is missing metric(s): {missing}"
    return metrics


def _tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def _skip_without_tensorflow(model_name: str) -> None:
    if model_name in {"lstm", "gru"} and not _tensorflow_available():
        pytest.skip(f"TensorFlow is required to load and predict with {model_name}.")


def _load_predictions(model_name: str, time_hhmm: str, dayofweek: int) -> pd.DataFrame:
    _skip_without_tensorflow(model_name)

    if model_name == "random_forest":
        from src.models.random_forest_model_v1 import predict_for_edges

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return predict_for_edges(
                model_path=str(_first_existing(MODEL_CONFIG[model_name]["model"], "RF model")),
                edges_path=str(EDGES_PATH),
                time_hhmm=time_hhmm,
                dayofweek=dayofweek,
            )

    if model_name == "lstm":
        from src.models.lstm_model_v4 import predict_for_edges

        return predict_for_edges(
            model_path=str(_first_existing(MODEL_CONFIG[model_name]["model"], "LSTM model")),
            metadata_path=str(_first_existing(MODEL_CONFIG[model_name]["metadata"], "LSTM metadata")),
            data_path=str(DATA_PATH),
            edges_path=str(EDGES_PATH),
            time_hhmm=time_hhmm,
            dayofweek=dayofweek,
        )

    if model_name == "gru":
        from src.models.gru_model_v4 import predict_for_edges

        return predict_for_edges(
            model_path=str(_first_existing(MODEL_CONFIG[model_name]["model"], "GRU model")),
            metadata_path=str(_first_existing(MODEL_CONFIG[model_name]["metadata"], "GRU metadata")),
            data_path=str(DATA_PATH),
            edges_path=str(EDGES_PATH),
            time_hhmm=time_hhmm,
            dayofweek=dayofweek,
        )

    raise ValueError(f"Unknown model: {model_name}")


@pytest.mark.parametrize("model_name", MODEL_CONFIG)
def test_saved_metrics_meet_acceptance_thresholds(model_name: str) -> None:
    metrics_path = _first_existing(MODEL_CONFIG[model_name]["metrics"], f"{model_name} metrics")
    metrics = _parse_metrics(metrics_path)

    assert metrics["MAE"] < METRIC_THRESHOLDS["MAE"], (
        f"{model_name} MAE should be below {METRIC_THRESHOLDS['MAE']}, "
        f"got {metrics['MAE']}"
    )
    assert metrics["RMSE"] < METRIC_THRESHOLDS["RMSE"], (
        f"{model_name} RMSE should be below {METRIC_THRESHOLDS['RMSE']}, "
        f"got {metrics['RMSE']}"
    )
    assert metrics["R2"] > METRIC_THRESHOLDS["R2"], (
        f"{model_name} R2 should be above {METRIC_THRESHOLDS['R2']}, "
        f"got {metrics['R2']}"
    )


def test_random_forest_model_artifact_loads() -> None:
    model_path = _first_existing(MODEL_CONFIG["random_forest"]["model"], "RF model")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(model_path)

    assert "pipeline" in bundle
    assert callable(getattr(bundle["pipeline"], "predict", None))


@pytest.mark.parametrize("model_name", ["lstm", "gru"])
def test_sequence_model_artifact_loads(model_name: str) -> None:
    _skip_without_tensorflow(model_name)

    import tensorflow as tf

    model_path = _first_existing(MODEL_CONFIG[model_name]["model"], f"{model_name} model")
    if model_name == "gru":
        from src.models.gru_wrapper_v4 import GRURegressorWrapperV4

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"GRURegressor": GRURegressorWrapperV4},
            compile=False,
        )
    else:
        model = tf.keras.models.load_model(model_path, compile=False)

    assert callable(getattr(model, "predict", None))


@pytest.mark.parametrize("model_name", MODEL_CONFIG)
def test_edge_predictions_have_expected_schema_and_realistic_range(model_name: str) -> None:
    predictions = _load_predictions(model_name, time_hhmm="0800", dayofweek=0)

    assert EXPECTED_PREDICTION_COLUMNS.issubset(predictions.columns)
    assert len(predictions) == len(pd.read_csv(EDGES_PATH))
    assert predictions[PREDICTION_COLUMN].notna().all()
    assert (predictions[PREDICTION_COLUMN] >= 0).all()
    assert (predictions[PREDICTION_COLUMN] < 2000).all()


def test_cross_model_monday_morning_predictions_are_not_identical() -> None:
    if not _tensorflow_available():
        pytest.skip("TensorFlow is required for the LSTM and GRU cross-model spot-check.")

    predictions = {
        model_name: _load_predictions(model_name, time_hhmm="0800", dayofweek=0)
        .sort_values("movement_id")[PREDICTION_COLUMN]
        .to_numpy()
        for model_name in MODEL_CONFIG
    }

    assert not np.allclose(predictions["random_forest"], predictions["lstm"])
    assert not np.allclose(predictions["random_forest"], predictions["gru"])
    assert not np.allclose(predictions["lstm"], predictions["gru"])


def test_peak_hour_average_is_higher_than_off_peak_for_at_least_two_models() -> None:
    if not _tensorflow_available():
        pytest.skip("TensorFlow is required for the full MT-06 model comparison.")

    passing_models = []
    for model_name in MODEL_CONFIG:
        peak = _load_predictions(model_name, time_hhmm="0800", dayofweek=0)[PREDICTION_COLUMN].mean()
        off_peak = _load_predictions(model_name, time_hhmm="0200", dayofweek=0)[PREDICTION_COLUMN].mean()
        if peak > off_peak:
            passing_models.append(model_name)

    assert len(passing_models) >= 2, (
        "Expected at least two models to predict higher Monday 08:00 traffic "
        f"than Monday 02:00 traffic; passing models: {passing_models}"
    )


def test_weekday_average_is_higher_than_weekend_for_at_least_two_models() -> None:
    if not _tensorflow_available():
        pytest.skip("TensorFlow is required for the full MT-07 model comparison.")

    passing_models = []
    for model_name in MODEL_CONFIG:
        weekday = _load_predictions(model_name, time_hhmm="0900", dayofweek=1)[PREDICTION_COLUMN].mean()
        weekend = _load_predictions(model_name, time_hhmm="0900", dayofweek=6)[PREDICTION_COLUMN].mean()
        if weekday > weekend:
            passing_models.append(model_name)

    assert len(passing_models) >= 2, (
        "Expected at least two models to predict higher Tuesday 09:00 traffic "
        f"than Sunday 09:00 traffic; passing models: {passing_models}"
    )
