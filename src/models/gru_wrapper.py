import os
import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.models.gru_model_v1 import GRURegressor as CustomGRURegressor, train_gru


class GRURegressorWrapper:
    def __init__(
        self,
        model_path="models_saved/gru_model_v1.keras",
        metadata_path="models_saved/gru_metadata_v1.joblib",
        data_path="data/processed/traffic_data.csv",
        edges_path="data/processed/edges.csv",
    ):
        # Ensure traffic data and edges exist
        if not os.path.exists(data_path) or not os.path.exists(edges_path):
            raise FileNotFoundError("Required processed CSV data files not found.")

        # Load/Train keras model if missing
        if not os.path.exists(model_path):
            print(
                f"GRU model files not found. Automatically training GRU model now (this may take a few moments)..."
            )
            train_gru(
                data_path=data_path,
                output_path=model_path,
                metadata_output_path=metadata_path,
                metrics_output_path="models_saved/gru_metrics_v1.csv",
            )
            print("GRU model training completed successfully.")

        # Load metadata/scaler
        if os.path.exists(metadata_path):
            meta = joblib.load(metadata_path)
            self.scaler = meta["scaler"]
            self.window_size = meta.get("window_size", 12)
        else:
            fallback_scaler = "models_saved/scaler.joblib"
            if os.path.exists(fallback_scaler):
                self.scaler = joblib.load(fallback_scaler)
            else:
                print(f"Scaler not found at {
                      metadata_path}. Fitting a new MinMaxScaler on traffic data...")
                df_temp = pd.read_csv(data_path)
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.scaler.fit(df_temp[["hourly_traffic_volume"]])
                os.makedirs(os.path.dirname(fallback_scaler), exist_ok=True)
                joblib.dump(self.scaler, fallback_scaler)
            self.window_size = 12

        # Load keras model
        self.model = tf.keras.models.load_model(
            model_path, custom_objects={"GRURegressor": CustomGRURegressor}
        )

        # Detect feature columns dynamically based on Keras model input layer shape
        try:
            num_features = self.model.input_shape[-1]
        except AttributeError:
            try:
                num_features = self.model.layers[0].weights[0].shape[0]
            except Exception:
                num_features = 3
        if num_features == 2:
            self.feature_cols = ["scaled_volume", "scaled_hour"]
        else:
            self.feature_cols = ["scaled_volume", "scaled_hour", "scaled_dayofweek"]

        # Load traffic data
        self.traffic_df = pd.read_csv(data_path)
        self.traffic_df["DateTime"] = pd.to_datetime(
            self.traffic_df["DateTime"], dayfirst=True, errors="coerce"
        )
        self.traffic_df = self.traffic_df.dropna(subset=["DateTime"])
        self.traffic_df["hour"] = self.traffic_df["DateTime"].dt.hour.astype(int)
        self.traffic_df["movement_id"] = self.traffic_df["movement_id"].astype(str)
        # Scale features
        self.traffic_df["scaled_volume"] = self.scaler.transform(
            self.traffic_df[["hourly_traffic_volume"]]
        )
        self.traffic_df["scaled_hour"] = self.traffic_df["hour"] / 23.0
        self.traffic_df["scaled_dayofweek"] = self.traffic_df["dayofweek"] / 6.0

        # Load edges to map (from_site, to_site) -> movement_id
        edges_df = pd.read_csv(edges_path)
        self.edge_to_movement = {}
        for _, row in edges_df.iterrows():
            u = int(row["from_site"])
            v = int(row["to_site"])
            m_id = str(row["movement_id"])
            self.edge_to_movement[(u, v)] = m_id
            self.edge_to_movement[(v, u)] = m_id

    def predict(self, from_site: int, to_site: int, time_str: str) -> float:
        movement_id = self.edge_to_movement.get((from_site, to_site))
        if not movement_id:
            return 300.0

        hist = (
            self.traffic_df[self.traffic_df["movement_id"] == movement_id]
            .sort_values("DateTime")
            .reset_index(drop=True)
        )

        if len(hist) < self.window_size:
            return 300.0

        seq_features = hist[self.feature_cols].values[-self.window_size :]
        X = np.expand_dims(seq_features, axis=0)

        pred_scaled = self.model.predict(X, verbose=0)
        pred_flow = self.scaler.inverse_transform(pred_scaled).flatten()[0]
        return float(np.maximum(0.0, pred_flow))
