# src/models/__init__.py

import torch.nn as nn
from src.models.base_model import BaseModel
from src.models.feature_extractors.eeg_resnet18 import build_eeg_resnet18 # Import the factory function
from src.models.feature_extractors.mlp_extractor import MLPExtractor
from src.models.heads.classification_head import ClassificationHead
# from src.models.heads.regression_head import RegressionHead 
# from src.models.heads.uncertainty_regression_head import UncertaintyRegressionHead
from typing import Dict, Any, List

def get_feature_extractor(model_config: dict, preprocess_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to get a feature extractor instance based on model config
    and preprocessing type.
    """
    # Determine which feature extractor to use based on preprocessing config
    use_de_features = preprocess_config.get('feature_extraction', {}).get('enable', False) and \
                      preprocess_config['feature_extraction'].get('type') == 'de'

    # Get common embedding_dim from model_config
    embedding_dim = model_config['feature_extractor']['embedding_dim']

    # Get in_channels for EEG data
    in_channels_eeg = model_config['feature_extractor']['in_channels']

    if use_de_features:
        # Calculate input_dim for MLP: num_channels * num_bands
        num_bands = len(preprocess_config['feature_extraction'].get('bands', {}))
        if num_bands == 0:
            raise ValueError("Differential Entropy feature extraction enabled but no bands defined in preprocess config.")
        
        input_dim_for_mlp = in_channels_eeg * num_bands
        
        # Get MLP specific hidden dims
        mlp_hidden_dims = model_config['feature_extractor'].get('mlp_hidden_dims', [])
        
        extractor = MLPExtractor(
            input_dim=input_dim_for_mlp,
            hidden_dims=mlp_hidden_dims,
            output_dim=embedding_dim # MLP's output is the requested embedding_dim
        )
    else:
        # Get in_channels for EEGResNet18
        extractor = build_eeg_resnet18(
            in_channels=in_channels_eeg, # Use configured in_channels
            embedding_dim=embedding_dim
        )
    
    extractor.output_dim = embedding_dim 
    
    return extractor

def get_model_head(model_head_config: dict, feature_extractor_out_dim: int) -> nn.Module:
    """Factory function to get a model head (classification/regression)."""
    head_type = model_head_config['type'].lower()
    if head_type == 'classification':
        return ClassificationHead(
            input_dim=feature_extractor_out_dim, 
            num_classes=model_head_config['num_classes']
        )
    # elif head_type == 'regression':
    #     return RegressionHead(input_dim=feature_extractor_out_dim, output_dim=model_head_config['output_dim'])
    # elif head_type == 'uncertainty_regression':
    #     return UncertaintyRegressionHead(input_dim=feature_extractor_out_dim, output_dim=model_head_config['output_dim'])
    else:
        raise ValueError(f"Unsupported model head type: {head_type}")

def get_eeg_model(config: dict, preprocess_config: Dict[str, Any]) -> BaseModel:
    """
    Factory function to build the complete EEG model (feature extractor + head).
    `preprocess_config` is passed to dynamically determine feature extractor type.
    """
    # Create the feature extractor based on model and preprocessing configs
    feature_extractor = get_feature_extractor(config, preprocess_config) # Pass full config to allow access to model.feature_extractor params
    
    # Get the actual output dimension of the instantiated feature_extractor instance
    feature_extractor_out_dim = feature_extractor.output_dim
    
    # Create the model head
    model_head = get_model_head(config['model_head'], feature_extractor_out_dim)

    return BaseModel(feature_extractor, model_head)

