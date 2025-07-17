# src/models/heads/uncertainty_regression_head.py

import torch.nn as nn
import torch

class UncertaintyRegressionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2): # output_dim for Valence, Arousal
        super().__init__()
        self.mean_layer = nn.Linear(input_dim, output_dim)
        self.log_variance_layer = nn.Linear(input_dim, output_dim) # Output log(sigma^2) for non-negativity

    def forward(self, x: torch.Tensor):
        mu_pred = self.mean_layer(x)
        log_sigma_sq_pred = self.log_variance_layer(x) 
        return mu_pred, log_sigma_sq_pred
