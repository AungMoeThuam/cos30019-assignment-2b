#!/usr/bin/env python3
"""
GRU sequence model for TBRGS traffic flow prediction.
"""

from __future__ import annotations

import argparse
import datetime
import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from src.models.model_utils import (
    FEATURES,
    TARGET,
    hhmm_to_hour,
    validate_dayofweek,
    load_traffic_data,
    evaluate_model,
    prepare_sequences,
)


# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class GRURegressor(tf.keras.Model):
    def __init__(self, hidden_units=64, output_dim=1):
        super(GRURegressor, self).__init__()
        # GRU layer: receives [batch_size, seq_len, num_features]
        self.gru_layer = layers.GRU(hidden_units, return_sequences=False)
        # Dense layer: maps hidden state output to target dimension
        self.dense_layer = layers.Dense(output_dim)

    def call(self, x):
        x = self.gru_layer(x)
        return self.dense_layer(x)

    def get_config(self):
        return {"hidden_units": 64, "output_dim": 1}


def train_gru(
    data_path: str,
    output_path: str,
    metadata_output_path: str,
    metrics_output_path: str,
    window_size: int = 4,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """Train GRU model and serialize both weights and metadata."""
    df = load_traffic_data(data_path)
    X_train, y_train, X_test, y_test, scaler, cutoff, _, _ = prepare_sequences(
        df, window_size=window_size, train_ratio=train_ratio
    )

    model = GRURegressor(hidden_units=64, output_dim=1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    print("Training GRU model...")
    print(f"Training sequences: {len(X_train)}")
    print(f"Testing sequences : {len(X_test)}")
    print(f"Cutoff time        : {pd.to_datetime(cutoff)}")

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, shuffle=False)

    # Make predictions and inverse scale them
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = evaluate_model(y_test_original, predictions)

    # Save keras model weights
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)

    # Save metadata
    movement_to_scats = (
        df[["movement_id", "scats_number"]]
        .drop_duplicates()
        .set_index("movement_id")["scats_number"]
        .to_dict()
    )
    metadata = {
        "scaler": scaler,
        "features": FEATURES,
        "target": TARGET,
        "movement_to_scats": movement_to_scats,
        "metrics": metrics,
        "window_size": window_size,
    }
    os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
    joblib.dump(metadata, metadata_output_path)

    # Save metrics in simple key-value report format.
    metrics_text = (
        f"Model : GRURegressor\n"
        f"Target: {TARGET}\n"
        f"Features: {', '.join(FEATURES)}\n"
        f"Training Sequences: {len(X_train)}\n"
        f"Testing Sequences : {len(X_test)}\n"
        f"Cutoff Datetime: {pd.to_datetime(cutoff)}\n"
        f"MAE : {metrics['MAE']:.2f}\n"
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"R2  : {metrics['R2']:.4f}\n"
    )

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as file:
        file.write(metrics_text)

    print("\nGRU Evaluation")
    print(f"MAE : {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2  : {metrics['R2']:.4f}")
    print(f"\nSaved model     : {output_path}")
    print(f"Saved metadata  : {metadata_output_path}")
    print(f"Saved metrics   : {metrics_output_path}")

    return metadata


def predict_for_edges(
    model_path: str,
    metadata_path: str,
    data_path: str,
    edges_path: str,
    time_hhmm: str | int,
    dayofweek: int | None = None,
    output_csv_path: str | None = None,
) -> pd.DataFrame:
    """Predict traffic volume on edges using the trained GRU model."""
    meta = joblib.load(metadata_path)
    scaler = meta["scaler"]
    window_size = meta["window_size"]
    movement_to_scats = meta.get("movement_to_scats", {})

    model = tf.keras.models.load_model(model_path, custom_objects={"GRURegressor": GRURegressor})
    edges_df = pd.read_csv(edges_path)
    traffic_df = load_traffic_data(data_path)

    # Scale variables
    traffic_df["scaled_volume"] = scaler.transform(traffic_df[[TARGET]])
    traffic_df["scaled_hour"] = traffic_df["hour"] / 23.0
    traffic_df["scaled_dayofweek"] = traffic_df["dayofweek"] / 6.0

    feature_cols = ["scaled_volume", "scaled_hour", "scaled_dayofweek"]
    hour = hhmm_to_hour(time_hhmm)

    # Option A: Default to today's weekday index if not specified
    if dayofweek is None:
        dayofweek = datetime.datetime.now().weekday()

    dayofweek = validate_dayofweek(dayofweek)

    X_pred_list = []
    valid_edges = []

    for idx, edge in edges_df.iterrows():
        movement_id = str(edge["movement_id"])

        # Retrieve historical sequence context for this movement ID
        hist_group = (
            traffic_df[traffic_df["movement_id"] == movement_id]
            .sort_values("DateTime")
            .reset_index(drop=True)
        )

        if len(hist_group) < window_size:
            dummy_feature = np.zeros((window_size, len(feature_cols)))
            X_pred_list.append(dummy_feature)
        else:
            seq_features = hist_group[feature_cols].values[-window_size:]
            X_pred_list.append(seq_features)

        valid_edges.append(edge)

    X_pred = np.array(X_pred_list)
    predicted_scaled = model.predict(X_pred)
    predicted_flow = scaler.inverse_transform(predicted_scaled).flatten()
    predicted_flow = np.maximum(0, predicted_flow)

    result = pd.DataFrame(valid_edges)
    result["prediction_time_hhmm"] = str(time_hhmm).zfill(4)
    result["dayofweek"] = dayofweek
    result["predicted_hourly_traffic_volume"] = predicted_flow.round(2)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        result.to_csv(output_csv_path, index=False)
        print(f"Saved edge predictions: {output_csv_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="GRU sequence model for TBRGS traffic flow prediction."
    )

    parser.add_argument(
        "--mode",
        choices=["train", "predict_edges"],
        default="train",
        help="train = train GRU model, predict_edges = predict flow for every edge",
    )

    parser.add_argument(
        "--data",
        default="data/processed/traffic_data.csv",
        help="Path to processed traffic_data.csv",
    )

    parser.add_argument(
        "--edges",
        default="data/processed/edges.csv",
        help="Path to processed edges.csv",
    )

    parser.add_argument(
        "--model",
        default="models_saved/gru_model_v1.keras",
        help="Path to saved GRU Keras model",
    )

    parser.add_argument(
        "--out",
        default="models_saved/gru_model_v1.keras",
        help="Output path for trained Keras model weights",
    )

    parser.add_argument(
        "--metadata",
        default="models_saved/gru_metadata_v1.joblib",
        help="Output path for model scaling metadata",
    )

    parser.add_argument(
        "--metrics",
        default="models_saved/gru_metrics_v1.csv",
        help="Output path for model evaluation metrics",
    )

    parser.add_argument(
        "--time",
        default="1100",
        help="Prediction time in HHMM format, for example 1100",
    )

    parser.add_argument(
        "--dayofweek",
        type=int,
        default=None,
        help="Monday=0, Tuesday=1, ..., Sunday=6. Defaults to current day of the week if not specified.",
    )

    parser.add_argument(
        "--out_csv",
        default="data/processed/gru_edge_predictions_v1.csv",
        help="Output CSV path for edge predictions",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_gru(
            data_path=args.data,
            output_path=args.out,
            metadata_output_path=args.metadata,
            metrics_output_path=args.metrics,
        )

    elif args.mode == "predict_edges":
        predictions_df = predict_for_edges(
            model_path=args.model,
            metadata_path=args.metadata,
            data_path=args.data,
            edges_path=args.edges,
            time_hhmm=args.time,
            dayofweek=args.dayofweek,
            output_csv_path=args.out_csv,
        )

        print("\nPredicted vehicles/hour for edges:")
        display_cols = [
            "from_site",
            "to_site",
            "movement_id",
            "prediction_time_hhmm",
            "predicted_hourly_traffic_volume",
        ]
        print(predictions_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
