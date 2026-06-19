#!/usr/bin/env python3
"""
Shared utilities for TBRGS traffic flow prediction models.

This module provides constants and helper functions used across
gru_model_v1, lstm_model_v1, and random_forest_model_v1.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def prepare_sequences(df: pd.DataFrame, window_size: int = 4, train_ratio: float = 0.8):
    """Scale data and prepare chronological window sequences for sequence models (GRU, LSTM).

    Args:
        df: Traffic DataFrame returned by load_traffic_data.
        window_size: Number of past timesteps in each input sequence.
                     GRU uses 4 (default); LSTM uses 12.
        train_ratio: Fraction of unique timestamps allocated to the training set.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, scaler, cutoff,
                  test_dates, test_movements).
    """
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
        group_train = (
            train_df[train_df["movement_id"] == movement_id]
            .sort_values("DateTime")
            .reset_index(drop=True)
        )
        group_test = (
            test_df[test_df["movement_id"] == movement_id]
            .sort_values("DateTime")
            .reset_index(drop=True)
        )

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
        scaler, cutoff, test_dates, test_movements,
    )


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
