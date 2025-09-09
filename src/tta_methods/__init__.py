# src/tta_methods/__init__.py

import torch
import torch.nn as nn
from typing import Dict, Any

from src.tta_methods.base_tta import BaseTTAMethod
from src.tta_methods.tent_tta import Tent
from src.tta_methods.cotta_tta import CoTTA
from src.tta_methods.adj_matrix_matcha import AdjMatrixMatcha, NoOpTTA

# Centralized dictionary to map method names to their classes
TTA_METHOD_CLASSES = {
    'tent': Tent,
    'cotta': CoTTA,
    'adjmatrixmatcha': AdjMatrixMatcha,
    'nooptta': NoOpTTA,
}

def get_tta_method(model: nn.Module, config: Dict, device: torch.device) -> BaseTTAMethod:
    """
    Factory function to get an instantiated TTA method based on configuration.

    Args:
        model (nn.Module): The model to which the TTA method will be applied.
        config (Dict): Configuration dictionary specific to the TTA method.
        device (torch.device): The device (CPU/GPU) to run computations on.
    
    Returns:
        BaseTTAMethod: An instantiated TTA method.
    
    Raises:
        ValueError: If an unsupported TTA method is specified.
    """
    tta_config = config['tta']
    method_name = tta_config['method'].lower()

    method_class = TTA_METHOD_CLASSES.get(method_name)
    if method_class is None:
        raise ValueError(f"Unsupported TTA method: {method_name}. "
                         f"Available methods: {list(TTA_METHOD_CLASSES.keys())}")

    return method_class(model=model, config=config, device=device)

