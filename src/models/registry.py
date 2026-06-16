# src/models/registry.py
# This module registers available models for dependency injection.

from src.models.lstm_model import LSTMRegressor
from src.models.gru_model import GRURegressor
from src.models.random_model import RandomBaselineModel

MODEL_REGISTRY = {
    "LSTM": LSTMRegressor,
    "GRU": GRURegressor,
    "RANDOM": RandomBaselineModel,
}

def get_model(name: str):
    """
    Return the model class corresponding to the given name.
    Raises ValueError if the name is not recognised.

    :param name: model name string (e.g. 'LSTM', 'GRU')
    :return: model class (not an instance)
    """
    model_class = MODEL_REGISTRY.get(name.upper())
    if not model_class:
        raise ValueError(f"Model '{name}' is not recognized. Available: {list(MODEL_REGISTRY.keys())}")
    return model_class
