# src/models/feature_extractors/mlp_extractor.py

import torch
import torch.nn as nn
from typing import List

class MLPExtractor(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) feature extractor for 1D features like Differential Entropy.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(MLPExtractor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            # Optional: Add BatchNorm1d for MLP layers
            # if h_dim > 1: # BatchNorm1d needs at least 1 feature dimension
            #     layers.append(nn.BatchNorm1d(h_dim)) 
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input expected: (batch_size, input_dim) - a flattened feature vector
        return self.mlp(x)

