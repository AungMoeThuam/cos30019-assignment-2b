#!/usr/bin/env python3
"""
LSTM traffic-flow model for TBRGS.
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

FEATURES = ["movement_id", "scats_number", "dayofweek", "isweekend", "hour"]
TARGET = "hourly_traffic_volume"


def hhmm_to_hour(time_hhmm: str | int) -> int:
    """Convert HHMM input into hour (0-23)."""
    time_text = str(time_hhmm).strip().zfill(4)

    if len(time_text) != 4 or not time_text.isdigit():
        raise ValueError("Time must be in HHMM format, for example 1100 or 0830.")

    hour = int(time_text[:2])
    minute = int(time_text[2:])

    if hour < 0 or hour > 23:
        raise ValueError("Hour must be between 00 and 23.")

    if minute < 0 or minute > 59:
        raise ValueError("Minute must be between 00 and 59.")

    return hour


def validate_dayofweek(dayofweek: int) -> int:
    """Validate dayofweek index (Monday=0, Sunday=6)."""
    dayofweek = int(dayofweek)

    if dayofweek < 0 or dayofweek > 6:
        raise ValueError("dayofweek must be 0 to 6. Monday=0, Sunday=6.")

    return dayofweek


def load_traffic_data(csv_path: str) -> pd.DataFrame:
    """Load and clean processed traffic_data.csv."""
    df = pd.read_csv(csv_path)

    required_columns = [
        "movement_id",
        "scats_number",
        "DateTime",
        "dayofweek",
        "isweekend",
        "hourly_traffic_volume",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in traffic data: {missing}")

    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DateTime"])

    df["movement_id"] = df["movement_id"].astype(str)
    df["scats_number"] = df["scats_number"].astype(int)
    df["dayofweek"] = df["dayofweek"].astype(int)
    df["isweekend"] = df["isweekend"].astype(int)
    df[TARGET] = df[TARGET].astype(float)

    # Extract hour from DateTime.
    df["hour"] = df["DateTime"].dt.hour.astype(int)

    # Keep records in chronological order
    df = df.sort_values(["DateTime", "movement_id"]).reset_index(drop=True)

    return df


def prepare_sequences(df: pd.DataFrame, window_size: int = 12, train_ratio: float = 0.8):
    """Scale data and prepare chronological window sequences for LSTM."""
    # Chronological cutoff
    unique_times = np.sort(df["DateTime"].unique())
    cutoff = unique_times[int(len(unique_times) * train_ratio)]
    
    train_df = df[df["DateTime"] < cutoff].copy()
    test_df = df[df["DateTime"] >= cutoff].copy()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df["scaled_volume"] = scaler.fit_transform(train_df[[TARGET]])
    test_df["scaled_volume"] = scaler.transform(test_df[[TARGET]])
    
    # Scale hours (0-23) and day of week (0-6)
    train_df["scaled_hour"] = train_df["hour"] / 23.0
    test_df["scaled_hour"] = test_df["hour"] / 23.0
    train_df["scaled_dayofweek"] = train_df["dayofweek"] / 6.0
    test_df["scaled_dayofweek"] = test_df["dayofweek"] / 6.0
    
    feature_cols = ["scaled_volume", "scaled_hour", "scaled_dayofweek"]
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_dates = []
    test_movements = []
    
    # Group by movement ID to prevent cross-movement sequence contamination
    for movement_id, group in df.groupby("movement_id"):
        group_train = train_df[train_df["movement_id"] == movement_id].sort_values("DateTime").reset_index(drop=True)
        group_test = test_df[test_df["movement_id"] == movement_id].sort_values("DateTime").reset_index(drop=True)
        
        combined_group = pd.concat([group_train, group_test], axis=0).reset_index(drop=True)
        feature_vals = combined_group[feature_cols].values
        target_vals = combined_group["scaled_volume"].values
        dates_vals = combined_group["DateTime"].values
        
        for i in range(window_size, len(combined_group)):
            seq_x = feature_vals[i - window_size : i]
            seq_y = target_vals[i]
            date_curr = dates_vals[i]
            
            if date_curr < cutoff:
                X_train.append(seq_x)
                y_train.append(seq_y)
            else:
                X_test.append(seq_x)
                y_test.append(seq_y)
                test_dates.append(date_curr)
                test_movements.append(movement_id)
                
    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test),
        scaler, cutoff, test_dates, test_movements
    )


def build_lstm_model(input_shape) -> Sequential:
    """Build deep LSTM network."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def train_lstm(
    data_path: str,
    output_path: str,
    metadata_output_path: str,
    metrics_output_path: str,
    window_size: int = 12,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """Train LSTM model and serialize both weights and metadata."""
    df = load_traffic_data(data_path)
    X_train, y_train, X_test, y_test, scaler, cutoff, _, _ = prepare_sequences(
        df, window_size=window_size, train_ratio=train_ratio
    )
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    print("Training LSTM model...")
    print(f"Training sequences: {len(X_train)}")
    print(f"Testing sequences : {len(X_test)}")
    print(f"Cutoff time        : {pd.to_datetime(cutoff)}")
    
    model.fit(X_train, y_train, epochs=25, batch_size=64, validation_split=0.1, shuffle=False)
    
    # Make predictions and inverse scale them
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    metrics = evaluate_model(y_test_original, predictions)

    # Save keras model
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
        f"Model : LSTMRegressor\n"
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

    print("\nLSTM Evaluation")
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
    """Predict traffic volume on edges using the trained LSTM model."""
    meta = joblib.load(metadata_path)
    scaler = meta["scaler"]
    window_size = meta["window_size"]
    movement_to_scats = meta.get("movement_to_scats", {})

    model = tf.keras.models.load_model(model_path)
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
        hist_group = traffic_df[traffic_df["movement_id"] == movement_id].sort_values("DateTime").reset_index(drop=True)
        
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
        description="LSTM sequence model for TBRGS traffic flow prediction."
    )

    parser.add_argument(
        "--mode",
        choices=["train", "predict_edges"],
        default="train",
        help="train = train LSTM model, predict_edges = predict flow for every edge",
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
        default="models_saved/lstm_model_v1.keras",
        help="Path to saved LSTM Keras model",
    )

    parser.add_argument(
        "--out",
        default="models_saved/lstm_model_v1.keras",
        help="Output path for trained Keras model weights",
    )

    parser.add_argument(
        "--metadata",
        default="models_saved/lstm_metadata_v1.joblib",
        help="Output path for model scaling metadata",
    )

    parser.add_argument(
        "--metrics",
        default="models_saved/lstm_metrics_v1.csv",
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
        default="data/processed/lstm_edge_predictions_v1.csv",
        help="Output CSV path for edge predictions",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_lstm(
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
