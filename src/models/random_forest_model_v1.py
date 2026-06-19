#!/usr/bin/env python3
"""
Random Forest traffic-flow model for TBRGS.
"""

from __future__ import annotations

import argparse
import datetime
import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.model_utils import (
    FEATURES,
    TARGET,
    hhmm_to_hour,
    validate_dayofweek,
    load_traffic_data,
    evaluate_model,
)


def build_random_forest_pipeline() -> Pipeline:
    """Create preprocessing + Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("movement", OneHotEncoder(handle_unknown="ignore"), ["movement_id"]),
            ("numeric", "passthrough", ["scats_number", "dayofweek", "isweekend", "hour"]),
        ]
    )

    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf_model),
        ]
    )

    return pipeline


def time_based_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split data chronologically by DateTime."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    unique_times = np.sort(df["DateTime"].unique())

    if len(unique_times) < 2:
        raise ValueError("Not enough unique DateTime values for train/test split.")

    cutoff = unique_times[int(len(unique_times) * train_ratio)]

    train_df = df[df["DateTime"] < cutoff].copy()
    test_df = df[df["DateTime"] >= cutoff].copy()

    return train_df, test_df, cutoff


def train_random_forest(
    data_path: str,
    output_path: str,
    metrics_output_path: str,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """Train Random Forest model and serialize pipeline/metadata."""
    df = load_traffic_data(data_path)

    train_df, test_df, cutoff = time_based_train_test_split(df, train_ratio=train_ratio)

    pipeline = build_random_forest_pipeline()

    print("Training Random Forest model...")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")
    print(f"Cutoff time  : {pd.to_datetime(cutoff)}")

    pipeline.fit(train_df[FEATURES], train_df[TARGET])

    predictions = pipeline.predict(test_df[FEATURES])
    metrics = evaluate_model(test_df[TARGET], predictions)

    movement_to_scats = (
        df[["movement_id", "scats_number"]]
        .drop_duplicates()
        .set_index("movement_id")["scats_number"]
        .to_dict()
    )

    bundle = {
        "pipeline": pipeline,
        "features": FEATURES,
        "target": TARGET,
        "movement_to_scats": movement_to_scats,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)

    # Save metrics in simple key-value report format.
    metrics_text = (
        f"Model : RandomForestRegressor\n"
        f"Target: {TARGET}\n"
        f"Features: {', '.join(FEATURES)}\n"
        f"Training Rows: {len(train_df)}\n"
        f"Testing Rows : {len(test_df)}\n"
        f"Cutoff Datetime: {pd.to_datetime(cutoff)}\n"
        f"MAE : {metrics['MAE']:.2f}\n"
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"R2  : {metrics['R2']:.4f}\n"
    )

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    with open(metrics_output_path, "w", encoding="utf-8") as file:
        file.write(metrics_text)

    print("\nRandom Forest Evaluation")
    print(f"MAE : {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2  : {metrics['R2']:.4f}")

    print(f"\nSaved model   : {output_path}")
    print(f"Saved metrics : {metrics_output_path}")

    return bundle


def predict_for_edges(
    model_path: str,
    edges_path: str,
    time_hhmm: str | int,
    dayofweek: int | None = None,
    output_csv_path: str | None = None,
) -> pd.DataFrame:
    """Predict traffic volume for all road edges."""
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    movement_to_scats = bundle.get("movement_to_scats", {})

    edges_df = pd.read_csv(edges_path)

    required_edge_columns = ["from_site", "to_site", "movement_id"]
    missing = [col for col in required_edge_columns if col not in edges_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in edges data: {missing}")

    hour = hhmm_to_hour(time_hhmm)

    # Option A: Default to today's weekday index if not specified
    if dayofweek is None:
        dayofweek = datetime.datetime.now().weekday()

    dayofweek = validate_dayofweek(dayofweek)
    isweekend = 1 if dayofweek in [5, 6] else 0

    prediction_rows = []

    for _, edge in edges_df.iterrows():
        movement_id = str(edge["movement_id"])
        scats_number = movement_to_scats.get(movement_id, edge["from_site"])

        prediction_rows.append(
            {
                "movement_id": movement_id,
                "scats_number": int(scats_number),
                "dayofweek": int(dayofweek),
                "isweekend": int(isweekend),
                "hour": int(hour),
            }
        )

    X_pred = pd.DataFrame(prediction_rows)
    predicted_flow = pipeline.predict(X_pred)
    predicted_flow = np.maximum(0, predicted_flow)

    result = edges_df.copy()
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
        description="Random Forest model for TBRGS traffic flow prediction."
    )

    parser.add_argument(
        "--mode",
        choices=["train", "predict_edges"],
        default="train",
        help="train = train RF model, predict_edges = predict flow for every edge",
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
        default="models_saved/random_forest_model_v1.pkl",
        help="Path to saved Random Forest model",
    )

    parser.add_argument(
        "--out",
        default="models_saved/random_forest_model_v1.pkl",
        help="Output path for trained Random Forest model",
    )

    parser.add_argument(
        "--metrics",
        default="models_saved/random_forest_metrics_v1.csv",
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
        default="data/processed/random_forest_edge_predictions_v1.csv",
        help="Output CSV path for edge predictions",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_random_forest(
            data_path=args.data,
            output_path=args.out,
            metrics_output_path=args.metrics,
        )

    elif args.mode == "predict_edges":
        predictions_df = predict_for_edges(
            model_path=args.model,
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
