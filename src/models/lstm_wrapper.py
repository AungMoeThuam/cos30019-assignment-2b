import os
import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.models.lstm_model_v1 import train_lstm


class LSTMRegressorWrapper:
    def __init__(
        self,
        model_path="models_saved/lstm_model_v1.keras",
        metadata_path="models_saved/lstm_metadata_v1.joblib",
        data_path="data/processed/traffic_data.csv",
        edges_path="data/processed/edges.csv",
    ):
        # Ensure traffic data and edges exist
        if not os.path.exists(data_path) or not os.path.exists(edges_path):
            raise FileNotFoundError("Required processed CSV data files not found.")

        # Load/Fit scaler
        if os.path.exists(metadata_path):
            meta = joblib.load(metadata_path)
            self.scaler = meta["scaler"]
        else:
            fallback_scaler = "models_saved/scaler.joblib"
            if os.path.exists(fallback_scaler):
                self.scaler = joblib.load(fallback_scaler)
            else:
                print("Metadata or scaler not found. Fitting a new MinMaxScaler on traffic data...")
                df_temp = pd.read_csv(data_path)
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.scaler.fit(df_temp[["hourly_traffic_volume"]])
                os.makedirs(os.path.dirname(fallback_scaler), exist_ok=True)
                joblib.dump(self.scaler, fallback_scaler)
                print(f"Saved new scaler to {fallback_scaler}")

        # Load/Train keras model
        if not os.path.exists(model_path) and not os.path.exists(
            "models_saved/lstm_model_v1.keras"
        ):
            print(
                f"LSTM model files not found. Automatically training LSTM model now (this may take a few moments)..."
            )
            train_lstm(
                data_path=data_path,
                output_path="models_saved/lstm_model_v1.keras",
                metadata_output_path="models_saved/lstm_metadata_v1.joblib",
                metrics_output_path="models_saved/lstm_metrics_v1.csv",
            )
            print("LSTM model training completed successfully.")

        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = tf.keras.models.load_model("models_saved/lstm_model_v1.keras")

        # Detect feature columns dynamically based on Keras model input layer shape
        # The input shape format is: (None, window_size, num_features)
        num_features = self.model.input_shape[-1]
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

        self.window_size = 12

    def predict(self, from_site: int, to_site: int, time_str: str) -> float:
        movement_id = self.edge_to_movement.get((from_site, to_site))
        if not movement_id:
            return 300.0

        # Filter historical data for this movement_id
        hist = (
            self.traffic_df[self.traffic_df["movement_id"] == movement_id]
            .sort_values("DateTime")
            .reset_index(drop=True)
        )

        if len(hist) < self.window_size:
            return 300.0

        # Extract the last window_size records
        seq_features = hist[self.feature_cols].values[-self.window_size :]

        # Reshape for Keras input: (1, window_size, num_features)
        X = np.expand_dims(seq_features, axis=0)

        # Predict
        pred_scaled = self.model.predict(X, verbose=0)
        pred_flow = self.scaler.inverse_transform(pred_scaled).flatten()[0]
        return float(np.maximum(0.0, pred_flow))
