#!/usr/bin/env python3
"""
GRU sequence model for TBRGS traffic flow prediction — v4.

Changes from v3:
  - movement_id handled via a Keras Embedding layer (dim=8) instead of
    label-encoding, giving the model a proper learned per-edge identity
    equivalent to RF's one-hot encoding.
  - scats_number added as a 5th numeric sequence feature.
  - Uses Keras Functional API (two-input model: numeric sequence + movement_id int).
  - Per-step features: [scaled_volume, scaled_hour, scaled_dayofweek,
    scaled_isweekend, scaled_scats_number] (5) + embedding (8) = 13 total.
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
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, Embedding, Concatenate,
    Reshape, RepeatVector,
)
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

from src.models.model_utils import (
    FEATURES,
    TARGET,
    hhmm_to_hour,
    validate_dayofweek,
    load_traffic_data,
    evaluate_model,
)


np.random.seed(42)
tf.random.set_seed(42)

EMBEDDING_DIM = 8


def build_gru_model(window_size: int, n_numeric: int,
                    n_movements: int, embedding_dim: int = EMBEDDING_DIM) -> Model:
    """Two-input GRU: numeric sequence + movement_id embedding."""
    seq_input = Input(shape=(window_size, n_numeric), name="numeric_sequence")
    mov_input = Input(shape=(1,), dtype="int32", name="movement_id")

    # movement_id → learned dense vector → tile across every time step
    emb = Embedding(n_movements, embedding_dim, name="movement_embedding")(mov_input)
    emb = Reshape((embedding_dim,))(emb)          # (batch, embedding_dim)
    emb = RepeatVector(window_size)(emb)           # (batch, window_size, embedding_dim)

    # Concatenate numeric features with embedding at each time step
    combined = Concatenate(axis=-1)([seq_input, emb])  # (batch, window_size, n_numeric+embedding_dim)

    x = GRU(128, return_sequences=False)(combined)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1)(x)

    model = Model(inputs=[seq_input, mov_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _build_sequences(df: pd.DataFrame, numeric_cols: list, target_col: str,
                     movement_to_int: dict, window_size: int, cutoff):
    """Build windowed sequences grouped by movement_id, returning movement_id ints."""
    train_df = df[df["DateTime"] < cutoff]
    test_df  = df[df["DateTime"] >= cutoff]

    X_num_tr, X_mov_tr, y_tr = [], [], []
    X_num_te, X_mov_te, y_te = [], [], []

    for movement_id, _ in df.groupby("movement_id"):
        mov_int = movement_to_int.get(movement_id, 0)
        g_tr = train_df[train_df["movement_id"] == movement_id].sort_values("DateTime").reset_index(drop=True)
        g_te = test_df[test_df["movement_id"]   == movement_id].sort_values("DateTime").reset_index(drop=True)
        combined = pd.concat([g_tr, g_te]).reset_index(drop=True)
        fv = combined[numeric_cols].values
        tv = combined[target_col].values
        dv = combined["DateTime"].values
        for i in range(window_size, len(combined)):
            seq_x = fv[i - window_size : i]
            seq_y = tv[i]
            if dv[i] < cutoff:
                X_num_tr.append(seq_x); X_mov_tr.append([mov_int]); y_tr.append(seq_y)
            else:
                X_num_te.append(seq_x); X_mov_te.append([mov_int]); y_te.append(seq_y)

    return (np.array(X_num_tr), np.array(X_mov_tr), np.array(y_tr),
            np.array(X_num_te), np.array(X_mov_te), np.array(y_te))


def train_gru(
    data_path: str,
    output_path: str,
    metadata_output_path: str,
    metrics_output_path: str,
    window_size: int = 4,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """Train GRU v4 with movement_id embedding + scats_number."""
    df = load_traffic_data(data_path)

    # Chronological split
    unique_times = np.sort(df["DateTime"].unique())
    cutoff = unique_times[int(len(unique_times) * train_ratio)]
    train_df = df[df["DateTime"] < cutoff].copy()
    test_df  = df[df["DateTime"] >= cutoff].copy()

    # Volume scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df["scaled_volume"] = scaler.fit_transform(train_df[[TARGET]])
    test_df["scaled_volume"]  = scaler.transform(test_df[[TARGET]])

    # scats_number normalisation
    max_scats = float(df["scats_number"].max())
    for _df in (train_df, test_df):
        _df["scaled_hour"]         = _df["hour"] / 23.0
        _df["scaled_dayofweek"]    = _df["dayofweek"] / 6.0
        _df["scaled_isweekend"]    = _df["isweekend"].astype(float)
        _df["scaled_scats_number"] = _df["scats_number"] / max_scats

    # movement_id integer encoding
    movement_ids    = sorted(df["movement_id"].unique())
    movement_to_int = {m: i for i, m in enumerate(movement_ids)}
    n_movements     = len(movement_ids)

    df = pd.concat([train_df, test_df]).sort_values(["DateTime", "movement_id"]).reset_index(drop=True)

    numeric_cols = [
        "scaled_volume",
        "scaled_hour",
        "scaled_dayofweek",
        "scaled_isweekend",
        "scaled_scats_number",
    ]

    X_num_tr, X_mov_tr, y_tr, X_num_te, X_mov_te, y_te = _build_sequences(
        df, numeric_cols, "scaled_volume", movement_to_int, window_size, cutoff
    )

    model = build_gru_model(window_size, len(numeric_cols), n_movements, EMBEDDING_DIM)

    print("Training GRU v4 model...")
    print(f"Training sequences : {len(X_num_tr)}")
    print(f"Testing sequences  : {len(X_num_te)}")
    print(f"Cutoff time        : {pd.to_datetime(cutoff)}")
    print(f"Numeric features   : {numeric_cols}")
    print(f"Embedding          : movement_id → dim {EMBEDDING_DIM}  ({n_movements} unique)")

    model.fit(
        [X_num_tr, X_mov_tr], y_tr,
        epochs=80, batch_size=64, validation_split=0.1, shuffle=False,
    )

    pred_scaled   = model.predict([X_num_te, X_mov_te])
    predictions   = scaler.inverse_transform(pred_scaled).flatten()
    y_te_original = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
    metrics       = evaluate_model(y_te_original, predictions)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)

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
        "numeric_cols": numeric_cols,
        "movement_to_int": movement_to_int,
        "n_movements": n_movements,
        "max_scats": max_scats,
        "embedding_dim": EMBEDDING_DIM,
    }
    os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
    joblib.dump(metadata, metadata_output_path)

    metrics_text = (
        f"Model : GRURegressor\n"
        f"Target: {TARGET}\n"
        f"Features: {', '.join(numeric_cols)} + movement_id_embedding(dim={EMBEDDING_DIM})\n"
        f"Training Sequences: {len(X_num_tr)}\n"
        f"Testing Sequences : {len(X_num_te)}\n"
        f"Cutoff Datetime: {pd.to_datetime(cutoff)}\n"
        f"MAE : {metrics['MAE']:.2f}\n"
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"R2  : {metrics['R2']:.4f}\n"
    )
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)

    print("\nGRU v4 Evaluation")
    print(f"MAE : {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2  : {metrics['R2']:.4f}")
    print(f"\nSaved model    : {output_path}")
    print(f"Saved metadata : {metadata_output_path}")
    print(f"Saved metrics  : {metrics_output_path}")
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
    """Predict traffic volume on edges using the trained GRU v4 model."""
    meta            = joblib.load(metadata_path)
    scaler          = meta["scaler"]
    window_size     = meta["window_size"]
    numeric_cols    = meta.get("numeric_cols", ["scaled_volume", "scaled_hour", "scaled_dayofweek"])
    movement_to_int = meta.get("movement_to_int", {})
    n_movements     = meta.get("n_movements", 1)
    max_scats       = meta.get("max_scats", 1.0)

    model      = tf.keras.models.load_model(model_path)
    edges_df   = pd.read_csv(edges_path)
    traffic_df = load_traffic_data(data_path)

    traffic_df["scaled_volume"]       = scaler.transform(traffic_df[[TARGET]])
    traffic_df["scaled_hour"]         = traffic_df["hour"] / 23.0
    traffic_df["scaled_dayofweek"]    = traffic_df["dayofweek"] / 6.0
    traffic_df["scaled_isweekend"]    = traffic_df["isweekend"].astype(float)
    traffic_df["scaled_scats_number"] = traffic_df["scats_number"] / max_scats

    hour = hhmm_to_hour(time_hhmm)
    if dayofweek is None:
        dayofweek = datetime.datetime.now().weekday()
    dayofweek = validate_dayofweek(dayofweek)
    isweekend = 1.0 if dayofweek >= 5 else 0.0

    scaled_hour_query = hour / 23.0
    scaled_dow_query  = dayofweek / 6.0

    X_num_list  = []
    X_mov_list  = []
    valid_edges = []

    for _, edge in edges_df.iterrows():
        movement_id = str(edge["movement_id"])
        mov_int     = movement_to_int.get(movement_id, 0)

        hist_group = (
            traffic_df[traffic_df["movement_id"] == movement_id]
            .sort_values("DateTime")
            .reset_index(drop=True)
        )

        target_mask = (
            (hist_group["hour"] == hour) &
            (hist_group["dayofweek"] == dayofweek)
        )
        target_positions = hist_group.index[target_mask].tolist()

        if target_positions and target_positions[-1] >= window_size:
            pos = target_positions[-1]
            seq_features = hist_group[numeric_cols].values[pos - window_size : pos].copy()
        elif target_positions:
            pos = target_positions[-1]
            available    = hist_group[numeric_cols].values[:pos]
            seq_features = np.zeros((window_size, len(numeric_cols)))
            if len(available) > 0:
                seq_features[-len(available):] = available
        else:
            if len(hist_group) < window_size:
                seq_features = np.zeros((window_size, len(numeric_cols)))
            else:
                seq_features = hist_group[numeric_cols].values[-window_size:].copy()
            col_map = {c: i for i, c in enumerate(numeric_cols)}
            seq_features[-1, col_map["scaled_hour"]]      = scaled_hour_query
            seq_features[-1, col_map["scaled_dayofweek"]] = scaled_dow_query
            if "scaled_isweekend" in col_map:
                seq_features[-1, col_map["scaled_isweekend"]] = isweekend

        X_num_list.append(seq_features)
        X_mov_list.append([mov_int])
        valid_edges.append(edge)

    X_num            = np.array(X_num_list)
    X_mov            = np.array(X_mov_list)
    predicted_scaled = model.predict([X_num, X_mov])
    predicted_flow   = scaler.inverse_transform(predicted_scaled).flatten()
    predicted_flow   = np.maximum(0, predicted_flow)

    result = pd.DataFrame(valid_edges)
    result["prediction_time_hhmm"]           = str(time_hhmm).zfill(4)
    result["dayofweek"]                       = dayofweek
    result["predicted_hourly_traffic_volume"] = predicted_flow.round(2)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        result.to_csv(output_csv_path, index=False)
        print(f"Saved edge predictions: {output_csv_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="GRU v4 — embedding + scats_number.")
    parser.add_argument("--mode",     choices=["train", "predict_edges"], default="train")
    parser.add_argument("--data",     default="data/processed/traffic_data.csv")
    parser.add_argument("--edges",    default="data/processed/edges.csv")
    parser.add_argument("--model",    default="models_saved/gru_model_v4.keras")
    parser.add_argument("--out",      default="models_saved/gru_model_v4.keras")
    parser.add_argument("--metadata", default="models_saved/gru_metadata_v4.joblib")
    parser.add_argument("--metrics",  default="models_saved/gru_metrics_v4.csv")
    parser.add_argument("--time",     default="1100")
    parser.add_argument("--dayofweek", type=int, default=None)
    parser.add_argument("--out_csv",  default="data/processed/gru_edge_predictions_v4.csv")
    args = parser.parse_args()

    if args.mode == "train":
        train_gru(
            data_path=args.data, output_path=args.out,
            metadata_output_path=args.metadata, metrics_output_path=args.metrics,
        )
    elif args.mode == "predict_edges":
        df = predict_for_edges(
            model_path=args.model, metadata_path=args.metadata,
            data_path=args.data, edges_path=args.edges,
            time_hhmm=args.time, dayofweek=args.dayofweek,
            output_csv_path=args.out_csv,
        )
        cols = ["from_site", "to_site", "movement_id",
                "prediction_time_hhmm", "predicted_hourly_traffic_volume"]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
