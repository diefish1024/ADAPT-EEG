# src/models/base_model.py

import torch
import torch.nn as nn

from src.models.feature_extractors.eeg_resnet18 import EEGResNet18 

from src.models.heads.classification_head import ClassificationHead

class BaseModel(nn.Module):
    """
    The top model that combines the feature extractor and model head.
    """
    def __init__(self, feature_extractor: nn.Module, model_head: nn.Module): # Modified signature
        """
        Args:
            feature_extractor (nn.Module): An instantiated feature extractor network.
            model_head (nn.Module): An instantiated model head network.
        """
        super(BaseModel, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.head = model_head

        print(f"Initialized BaseModel with Feature Extractor: {type(feature_extractor).__name__} and Head: {type(model_head).__name__}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): raw EEG input date, (batch_size, in_channels, time_points)。
        Returns:
            torch.Tensor: final output
        """
        features = self.feature_extractor(x)
        output = self.head(features)
        return output

