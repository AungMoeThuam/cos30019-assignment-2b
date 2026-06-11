# src/models/random_model.py
# This module defines a Random Baseline model.
# In machine learning, baseline models are used to prove that complex neural networks 
# (like LSTM and GRU) are performing better than simple random guessing.

import numpy as np

class RandomBaselineModel:
    # TODO:
    # 1. Define the __init__ method:
    #    - Store parameters like the minimum and maximum traffic flows observed in the training data.
    #    - Alternatively, store the mean and standard deviation of the training flow.
    
    def __init__(self, min_flow=0, max_flow=1500, mean_flow=None, std_flow=None):
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.mean_flow = mean_flow
        self.std_flow = std_flow
        pass

    # TODO:
    # 1. Define the predict method:
    #    - Accept sequential inputs (even if the random model ignores the sequence).
    #    - Return a random prediction.
    #    - Option A: Generate a random number between min_flow and max_flow using a uniform distribution:
    #      `predictions = np.random.uniform(self.min_flow, self.max_flow, size=num_samples)`
    #    - Option B: Generate a random number from a normal distribution using mean_flow and std_flow:
    #      `predictions = np.random.normal(self.mean_flow, self.std_flow, size=num_samples)`
    
    def predict(self, x):
        # x is the input sequence (batch_size, seq_len, num_features)
        # Return random flow predictions matching the batch size of x.
        pass

if __name__ == "__main__":
    # Test random model prediction output shape
    pass
