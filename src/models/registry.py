# src/models/registry.py
# This module registers available models for dependency injection.

from src.models.lstm_wrapper_v4 import LSTMRegressorWrapperV4
from src.models.gru_wrapper_v4 import GRURegressorWrapperV4
from src.models.random_forest_wrapper import RandomForestModelWrapper

MODEL_REGISTRY = {
    "LSTM": LSTMRegressorWrapperV4,
    "GRU": GRURegressorWrapperV4,
    "RANDOM": RandomForestModelWrapper,
    "RF": RandomForestModelWrapper,
    "RANDOM_FOREST": RandomForestModelWrapper,
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
