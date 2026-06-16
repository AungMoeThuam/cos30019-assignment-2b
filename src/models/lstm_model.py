import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
data = pd.read_csv('data/processed/traffic_data.csv')

# Clean column names
data.columns = (
    data.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Columns:", data.columns.tolist())

# Convert datetime
data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True)

# Sort by movement_id and datetime
data = data.sort_values(by=['movement_id', 'datetime']).reset_index(drop=True)

# Make sure traffic volume is numeric
data['hourly_traffic_volume'] = pd.to_numeric(
    data['hourly_traffic_volume'],
    errors='coerce'
)

# Drop missing values
data = data.dropna(subset=['movement_id', 'datetime', 'hourly_traffic_volume'])

# --------------------
# Feature Engineering
# --------------------

# Extract hour from datetime
data['hour'] = data['datetime'].dt.hour

# ----------------
# Scale features
# ----------------

scaler = MinMaxScaler(feature_range=(0, 1))
data['scaled_traffic_volume'] = scaler.fit_transform(data[['hourly_traffic_volume']])

# Scale hour only (0-23)
data['scaled_hour'] = data['hour'] / 23.0

# Using Volume and Hour as features
scaled_feature_cols = ['scaled_traffic_volume', 'scaled_hour']

# -----------------------------
# Create LSTM sequences per movement_id
# -----------------------------

window_size = 12   # previous 12 hours predict next hour
test_ratio = 0.15 # 85 - 15

X_train = []
y_train = []
X_test = []
y_test = []
dates_test = []
movement_test = []

for movement_id, group in data.groupby('movement_id'):
    group = group.sort_values('datetime').reset_index(drop=True)

    feature_values = group[scaled_feature_cols].values
    target_values = group['scaled_traffic_volume'].values
    dates = group['datetime'].values

    split_index = int(len(group) * (1 - test_ratio))

    for i in range(window_size, len(group)):
        X_window = feature_values[i - window_size:i]
        y_value = target_values[i]

        if i < split_index:
            X_train.append(X_window)
            y_train.append(y_value)
        else:
            X_test.append(X_window)
            y_test.append(y_value)
            dates_test.append(dates[i])
            movement_test.append(movement_id)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("X_train shape:", X_train.shape)

# Define the deeper model
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train model
history_tuned = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, shuffle=False)

# --------
# Predict
# --------

predictions_scaled = model.predict(X_test)

# Convert predictions back to real traffic volume
predictions_tuned = scaler.inverse_transform(predictions_scaled).flatten()

# Convert y_test back to real traffic volume
y_test_tuned = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

# -----------
# Evaluation
# -----------

rmse_tuned = np.sqrt(mean_squared_error(y_test_tuned, predictions_tuned))

print(f'RMSE after tuning: {rmse_tuned:.2f}')

mean_actual = np.mean(y_test_tuned)
print(f'Predicted traffic volume in test set: {mean_actual:.2f}')

if mean_actual > 0:
    percentage_error_tuned = (rmse_tuned / mean_actual) * 100
    print(f'RMSE as percentage of predicted traffic volume: {percentage_error_tuned:.2f}%')
else:
    print('Cannot calculate RMSE percentage because mean actual traffic volume is zero.')

# ----------------------
# Save Model and Scaler 
# ----------------------
import os
import joblib

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the Keras model
model.save('models_saved/lstm_model.keras')

# Save the scaler
joblib.dump(scaler, 'models_saved/scaler.joblib')


