# src/models/base_model.py

import torch
import torch.nn as nn
from src.utils.config_parser import load_config

from src.models.feature_extractors.resnet18 import EEGResNet18 

from src.models.heads.classification_head import ClassificationHead


class BaseModel(nn.Module):
    """
    The top model that combines the feature extractor and model head.
    Dynamically load and initialize the feature extractor and model head according to the configuration file.
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): A complete experiment configuration dictionary, including model-related configuration information.
                           config['model']['feature_extractor'] and config['model']['head']
        """
        super(BaseModel, self).__init__()
        self.config = config

        # initialize the feature extractor
        fe_cfg = config['model']['feature_extractor']
        if fe_cfg['name'] == 'ResNet18':
            self.feature_extractor = EEGResNet18(
                in_channels=fe_cfg['in_channels'],
                embedding_dim=fe_cfg['embedding_dim']
            )
        else:
            raise ValueError(f"Unknown feature extractor specified in config: {fe_cfg['name']}")

        # initialize the model header
        head_cfg = config['model']['head']
        head_input_dim = fe_cfg['embedding_dim'] 
        
        if head_cfg['type'] == 'classification':
            self.head = ClassificationHead(
                num_classes=head_cfg['num_classes'],
                embedding_dim=head_input_dim
            )
        else:
            raise ValueError(f"Unknown model head type specified in config: {head_cfg['type']}")

        print(f"Initialized BaseModel with Feature Extractor: {fe_cfg['name']} and Head: {head_cfg['type']}")

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

