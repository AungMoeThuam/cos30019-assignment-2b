import os
import datetime
import numpy as np
import pandas as pd
import joblib
from src.models.train_random_forest import train_random_forest

class RandomForestModelWrapper:
    def __init__(self, model_path="models_saved/random_forest_model.pkl", data_path="data/processed/traffic_data.csv", edges_path="data/processed/edges.csv"):
        # Ensure data paths exist
        if not os.path.exists(data_path) or not os.path.exists(edges_path):
            raise FileNotFoundError("Required processed CSV data files not found.")

        # Train pipeline if missing
        if not os.path.exists(model_path):
            print("Random Forest model file not found. Automatically training Random Forest model now (this may take a few moments)...")
            train_random_forest(
                data_path=data_path,
                output_path=model_path,
                metrics_output_path="models_saved/random_forest_metrics.csv"
            )
            print("Random Forest model training completed successfully.")

        # Load bundle
        try:
            self.bundle = joblib.load(model_path)
            self.pipeline = self.bundle["pipeline"]
            self.movement_to_scats = self.bundle.get("movement_to_scats", {})
        except Exception as e:
            print(f"Warning: Failed to load Random Forest model from {model_path} (likely due to scikit-learn version mismatch: {e}).")
            print("Retraining Random Forest model now to match local dependencies...")
            train_random_forest(
                data_path=data_path,
                output_path=model_path,
                metrics_output_path="models_saved/random_forest_metrics.csv"
            )
            self.bundle = joblib.load(model_path)
            self.pipeline = self.bundle["pipeline"]
            self.movement_to_scats = self.bundle.get("movement_to_scats", {})
            print("Random Forest model successfully retrained and loaded.")

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

        # Extract hour from time_str
        time_text = str(time_str).strip().zfill(4)
        hour = int(time_text[:2])

        # Get day of week
        dayofweek = datetime.datetime.now().weekday()
        isweekend = 1 if dayofweek in [5, 6] else 0

        # Get scats number
        scats_number = self.movement_to_scats.get(movement_id, from_site)

        # Form feature DataFrame for the prediction
        X = pd.DataFrame([{
            "movement_id": movement_id,
            "scats_number": int(scats_number),
            "dayofweek": int(dayofweek),
            "isweekend": int(isweekend),
            "hour": int(hour)
        }])

        predicted_flow = self.pipeline.predict(X)[0]
        return float(np.maximum(0.0, predicted_flow))
