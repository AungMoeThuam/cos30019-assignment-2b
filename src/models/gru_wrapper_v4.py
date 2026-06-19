import os
import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from src.models.model_utils import hhmm_to_hour, validate_dayofweek

class GRURegressorWrapperV4:
    def __init__(
        self,
        model_path="models_saved/gru_model_v4.keras",
        metadata_path="models_saved/gru_metadata_v4.joblib",
        data_path="data/processed/traffic_data.csv",
        edges_path="data/processed/edges.csv",
    ):
        if not os.path.exists(data_path) or not os.path.exists(edges_path):
            raise FileNotFoundError("Required processed CSV data files not found.")

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
        
        meta = joblib.load(metadata_path)
        self.scaler          = meta["scaler"]
        self.window_size     = meta["window_size"]
        self.numeric_cols    = meta.get("numeric_cols", ["scaled_volume", "scaled_hour", "scaled_dayofweek", "scaled_isweekend", "scaled_scats_number"])
        self.movement_to_int = meta.get("movement_to_int", {})
        self.n_movements     = meta.get("n_movements", 1)
        self.max_scats       = meta.get("max_scats", 1.0)

        # Load keras model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GRU v4 model not found at {model_path}.")
        self.model = tf.keras.models.load_model(model_path)

        # Load traffic data
        self.traffic_df = pd.read_csv(data_path)
        self.traffic_df["DateTime"] = pd.to_datetime(self.traffic_df["DateTime"], dayfirst=True, errors="coerce")
        self.traffic_df = self.traffic_df.dropna(subset=["DateTime"])
        
        # Calculate features matching v4 requirements
        self.traffic_df["hour"] = self.traffic_df["DateTime"].dt.hour.astype(int)
        self.traffic_df["dayofweek"] = self.traffic_df["DateTime"].dt.dayofweek.astype(int)
        self.traffic_df["isweekend"] = (self.traffic_df["dayofweek"] >= 5).astype(int)
        self.traffic_df["movement_id"] = self.traffic_df["movement_id"].astype(str)
        
        # Scale features
        self.traffic_df["scaled_volume"] = self.scaler.transform(self.traffic_df[["hourly_traffic_volume"]])
        self.traffic_df["scaled_hour"] = self.traffic_df["hour"] / 23.0
        self.traffic_df["scaled_dayofweek"] = self.traffic_df["dayofweek"] / 6.0
        self.traffic_df["scaled_isweekend"] = self.traffic_df["isweekend"].astype(float)
        self.traffic_df["scaled_scats_number"] = self.traffic_df["scats_number"] / self.max_scats

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

        mov_int = self.movement_to_int.get(movement_id, 0)
        hour = hhmm_to_hour(time_str)
        dayofweek = datetime.datetime.now().weekday()
        isweekend = 1.0 if dayofweek >= 5 else 0.0

        scaled_hour_query = hour / 23.0
        scaled_dow_query  = dayofweek / 6.0

        hist_group = (
            self.traffic_df[self.traffic_df["movement_id"] == movement_id]
            .sort_values("DateTime")
            .reset_index(drop=True)
        )

        target_mask = (
            (hist_group["hour"] == hour) &
            (hist_group["dayofweek"] == dayofweek)
        )
        target_positions = hist_group.index[target_mask].tolist()

        if target_positions and target_positions[-1] >= self.window_size:
            pos = target_positions[-1]
            seq_features = hist_group[self.numeric_cols].values[pos - self.window_size : pos].copy()
        elif target_positions:
            pos = target_positions[-1]
            available = hist_group[self.numeric_cols].values[:pos]
            seq_features = np.zeros((self.window_size, len(self.numeric_cols)))
            if len(available) > 0:
                seq_features[-len(available):] = available
        else:
            if len(hist_group) < self.window_size:
                seq_features = np.zeros((self.window_size, len(self.numeric_cols)))
            else:
                seq_features = hist_group[self.numeric_cols].values[-self.window_size:].copy()
            col_map = {c: i for i, c in enumerate(self.numeric_cols)}
            seq_features[-1, col_map["scaled_hour"]]      = scaled_hour_query
            seq_features[-1, col_map["scaled_dayofweek"]] = scaled_dow_query
            if "scaled_isweekend" in col_map:
                seq_features[-1, col_map["scaled_isweekend"]] = isweekend

        # Build dual inputs
        X_num = np.expand_dims(seq_features, axis=0)  # (1, window_size, 5)
        X_mov = np.array([[mov_int]])                 # (1, 1)

        # Predict with both inputs
        pred_scaled = self.model.predict([X_num, X_mov], verbose=0)
        pred_flow = self.scaler.inverse_transform(pred_scaled).flatten()[0]
        
        return max(0.0, float(pred_flow))
