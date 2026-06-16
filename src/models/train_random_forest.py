"""
Random Forest traffic-flow model for COS30019 Assignment 2B.

This file implements the third ML model: RandomForestRegressor.

It trains on:
    data/processed/traffic_data.csv

Expected traffic_data.csv columns:
    movement_id
    scats_number
    DateTime
    dayofweek
    isweekend
    hourly_traffic_volume

It can also predict traffic flow for every edge in:
    data/processed/edges.csv

Expected edges.csv columns:
    from_site
    to_site
    travel_distance_km
    movement_id

Main usage:

1) Train Random Forest:
    python src/models/train_random_forest.py --mode train \
        --data data/processed/traffic_data.csv \
        --out models_saved/random_forest_model.pkl \
        --metrics models_saved/random_forest_metrics.csv

2) Predict traffic flow for all edges at 11:00 AM on Monday:
    python src/models/train_random_forest.py --mode predict_edges \
        --model models_saved/random_forest_model.pkl \
        --edges data/processed/edges.csv \
        --time 1100 \
        --dayofweek 0 \
        --out_csv data/processed/rf_edge_predictions.csv

dayofweek format:
    Monday=0, Tuesday=1, Wednesday=2, Thursday=3,
    Friday=4, Saturday=5, Sunday=6
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURES = ["movement_id", "scats_number", "dayofweek", "isweekend", "hour"]
TARGET = "hourly_traffic_volume"


def hhmm_to_hour(time_hhmm: str | int) -> int:
    """
    Convert HHMM input into hour.

    Example:
        1100 -> 11
        0830 -> 8

    Random Forest in this implementation predicts hourly traffic volume,
    so the hour is enough as a model feature.
    """
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
    """Validate dayofweek where Monday=0 and Sunday=6."""
    dayofweek = int(dayofweek)

    if dayofweek < 0 or dayofweek > 6:
        raise ValueError("dayofweek must be 0 to 6. Monday=0, Sunday=6.")

    return dayofweek


def load_traffic_data(csv_path: str) -> pd.DataFrame:
    """
    Load and clean processed traffic_data.csv.

    This function prepares the exact columns needed by the Random Forest model.
    """
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

    # Keep records in time order to avoid data leakage during train/test split.
    df = df.sort_values(["DateTime", "movement_id"]).reset_index(drop=True)

    return df


def build_random_forest_pipeline() -> Pipeline:
    """
    Create the preprocessing + Random Forest pipeline.

    movement_id is categorical, so OneHotEncoder is used.
    Numeric columns are passed directly into the Random Forest model.
    """
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
    """
    Split data by DateTime.

    This is better than random split for time-series-style data because
    the model trains on earlier records and tests on later records.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    unique_times = np.sort(df["DateTime"].unique())

    if len(unique_times) < 2:
        raise ValueError("Not enough unique DateTime values for train/test split.")

    cutoff = unique_times[int(len(unique_times) * train_ratio)]

    train_df = df[df["DateTime"] < cutoff].copy()
    test_df = df[df["DateTime"] >= cutoff].copy()

    return train_df, test_df, cutoff


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """Return common regression metrics for report comparison."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def train_random_forest(
    data_path: str,
    output_path: str,
    metrics_output_path: str,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Train Random Forest and save it as a joblib bundle.

    The saved bundle contains:
        pipeline
        features
        target
        movement_to_scats
        metrics
    """
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


def load_model_bundle(model_path: str) -> Dict[str, Any]:
    """
    Load saved RF model bundle.

    Also supports older files that saved only the sklearn pipeline.
    """
    loaded = joblib.load(model_path)

    if isinstance(loaded, dict) and "pipeline" in loaded:
        return loaded

    # Backward compatibility if only pipeline was saved.
    return {
        "pipeline": loaded,
        "features": FEATURES,
        "target": TARGET,
        "movement_to_scats": {},
        "metrics": {},
    }


def predict_single_movement(
    model_path: str,
    movement_id: str | int,
    scats_number: int,
    time_hhmm: str | int,
    dayofweek: int = 0,
) -> float:
    """
    Predict hourly traffic volume for one movement/site at one time.
    """
    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]

    hour = hhmm_to_hour(time_hhmm)
    dayofweek = validate_dayofweek(dayofweek)
    isweekend = 1 if dayofweek in [5, 6] else 0

    X = pd.DataFrame(
        [
            {
                "movement_id": str(movement_id),
                "scats_number": int(scats_number),
                "dayofweek": int(dayofweek),
                "isweekend": int(isweekend),
                "hour": int(hour),
            }
        ]
    )

    prediction = pipeline.predict(X)[0]

    # Traffic volume cannot be negative.
    return float(max(0, prediction))


def predict_for_edges(
    model_path: str,
    edges_path: str,
    time_hhmm: str | int,
    dayofweek: int = 0,
    output_csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Predict hourly traffic flow for every edge in edges.csv.

    For each edge, this function uses:
        movement_id from edges.csv
        from_site as fallback scats_number if needed
        hour from HHMM
        dayofweek and isweekend
    """
    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    movement_to_scats = bundle.get("movement_to_scats", {})

    edges_df = pd.read_csv(edges_path)

    required_edge_columns = ["from_site", "to_site", "movement_id"]
    missing = [col for col in required_edge_columns if col not in edges_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in edges data: {missing}")

    hour = hhmm_to_hour(time_hhmm)
    dayofweek = validate_dayofweek(dayofweek)
    isweekend = 1 if dayofweek in [5, 6] else 0

    prediction_rows = []

    for _, edge in edges_df.iterrows():
        movement_id = str(edge["movement_id"])

        # Use the SCATS number learned during training.
        # If missing, fallback to edge's from_site.
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


def edge_predictions_to_flow_dict(predictions_df: pd.DataFrame) -> Dict[tuple, float]:
    """
    Convert edge prediction DataFrame into dictionary format for graph update.

    Output:
        {
            (from_site, to_site): predicted_flow,
            ...
        }
    """
    flow_dict = {}

    for _, row in predictions_df.iterrows():
        key = (int(row["from_site"]), int(row["to_site"]))
        flow_dict[key] = float(row["predicted_hourly_traffic_volume"])

    return flow_dict


def main():
    parser = argparse.ArgumentParser(
        description="Random Forest model for COS30019 Assignment 2B traffic flow prediction."
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
        default="models_saved/random_forest_model.pkl",
        help="Path to saved Random Forest model",
    )

    parser.add_argument(
        "--out",
        default="models_saved/random_forest_model.pkl",
        help="Output path for trained Random Forest model",
    )

    parser.add_argument(
        "--metrics",
        default="models_saved/random_forest_metrics.csv",
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
        default=0,
        help="Monday=0, Tuesday=1, ..., Sunday=6",
    )

    parser.add_argument(
        "--out_csv",
        default="data/processed/rf_edge_predictions.csv",
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
