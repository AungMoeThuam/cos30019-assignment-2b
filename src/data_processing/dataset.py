# src/data_processing/dataset.py
# This module is responsible for loading and preprocessing the SCATS traffic data.

def load_raw_data(file_path):
    # TODO:
    # 1. Open and read the raw Excel data file (e.g., Scats Data October 2006.xls) using pandas or xlrd.
    # 2. Extract intersection listings and coordinates from raw files.
    # 3. Clean any empty rows, handle invalid entries, and parse dates/times.
    pass

def preprocess_data(raw_data):
    # TODO:
    # 1. Aggregate traffic counts per 15-minute intervals.
    # 2. Deal with missing values (e.g., using linear interpolation or forward fill).
    # 3. Normalize the flow values (e.g., Min-Max scaling or Standard scaling) so they train better in LSTMs/GRUs.
    # 4. Extract temporal features:
    #    - Time of day (0 to 95 for 15-min intervals).
    #    - Day of week (0 to 6 for Monday to Sunday).
    pass

def generate_sequences(data, sequence_length):
    # TODO:
    # 1. Implement a sliding window approach to create sequences.
    #    - Input (X): Sequence of past flow measurements of size `sequence_length` (e.g., last 4 time steps).
    #    - Target (y): Flow measurement at the next time step.
    # 2. Split the generated sequences into Training and Testing sets (e.g., 80% train, 20% test).
    # 3. Save the preprocessed datasets into 'data/processed/' (e.g., as .pkl or PyTorch tensors) for fast loading.
    pass

if __name__ == "__main__":
    # Test script or execution runner for dataset creation
    pass
