import torch.nn as nn
from typing import Dict, Any

from src.models.base_model import BaseModel
from src.models.feature_extractors.eeg_resnet18 import build_eeg_resnet18
from src.models.feature_extractors.emt_cls import build_emt_extractor
from src.models.heads.classification_head import ClassificationHead
# from src.models.heads.regression_head import RegressionHead 
# from src.models.heads.uncertainty_regression_head import UncertaintyRegressionHead

def get_feature_extractor(model_config: dict, preprocess_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to get a feature extractor instance based on a type
    specified in the model configuration.

    Args:
        model_config: Configuration dictionary for the model.
        preprocess_config: Configuration dictionary for data preprocessing.

    Returns:
        An instance of the feature extractor (nn.Module).
    """
    extractor_config = model_config['feature_extractor']
    name = extractor_config['name']
    embedding_dim = extractor_config['embedding_dim']
    in_channels_eeg = extractor_config['in_channels']
    
    if name == 'emt_cls':
        feature_config = preprocess_config.get('feature_extraction', {})
        num_bands = len(feature_config.get('bands', {}))
        if num_bands == 0:
            raise ValueError("EMT model requires feature extraction (e.g., DE), but no frequency bands are defined in the preprocess config.")
        
        emt_params = extractor_config.get('emt_params', {})

        extractor = build_emt_extractor(
            in_channels=in_channels_eeg,
            num_features=num_bands,
            embedding_dim=embedding_dim,
            **emt_params
        )
    elif name == 'eeg_resnet18':
        extractor = build_eeg_resnet18(
            in_channels=in_channels_eeg,
            embedding_dim=embedding_dim
        )
    else:
        raise ValueError(f"Unsupported feature extractor: '{name}' specified in config.")
    
    extractor.output_dim = embedding_dim 
    
    return extractor

def get_model_head(model_head_config: dict, feature_extractor_out_dim: int) -> nn.Module:
    """
    Factory function to get a model head (e.g., for classification or regression).

    Args:
        model_head_config: Configuration for the model head.
        feature_extractor_out_dim: The output dimension of the feature extractor.

    Returns:
        An instance of the model head (nn.Module).
    """
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
    Factory function to build the complete EEG model by combining a feature 
    extractor and a model head.

    Args:
        config: The main configuration dictionary for the experiment.
        preprocess_config: The configuration for data preprocessing.

    Returns:
        A complete BaseModel instance.
    """
    # 1. Create the feature extractor based on model and preprocessing configs
    feature_extractor = get_feature_extractor(config['model'], preprocess_config)
    
    # 2. Get the actual output dimension from the instantiated feature extractor
    feature_extractor_out_dim = feature_extractor.output_dim
    
    # 3. Create the model head
    model_head = get_model_head(config['task']['model_head'], feature_extractor_out_dim)

    # 4. Combine them into the final model
    return BaseModel(feature_extractor, model_head)
