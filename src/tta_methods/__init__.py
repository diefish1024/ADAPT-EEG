# src/tta_methods/__init__.py

import torch
import torch.nn as nn
from typing import Dict, Any

from src.tta_methods.base_tta import BaseTTAMethod
from src.tta_methods.tent_tta import Tent

def get_tta_method(model: nn.Module, tta_config: Dict, device: torch.device) -> BaseTTAMethod:
    """
    Factory function to get an instantiated TTA method based on configuration.

    Args:
        model (nn.Module): The model to which the TTA method will be applied.
        tta_config (Dict): Configuration dictionary specific to the TTA method.
        device (torch.device): The device (CPU/GPU) to run computations on.
    
    Returns:
        BaseTTAMethod: An instantiated TTA method.
    
    Raises:
        ValueError: If an unsupported TTA method is specified.
    """
    method_name = tta_config['method'].lower()
    
    # Common parameters for TTA methods from config
    optimizer_config = tta_config.get('optimizer', {'type': 'adam', 'lr': 0.001})
    adapt_steps = tta_config.get('adapt_steps', 1)
    loss_weights = tta_config.get('loss_weights', {}) # Weights for specific TTA-related losses

    if method_name == 'Tent':
        # Tent constructor might need model, optimizer_config, adapt_steps
        return Tent(model=model, optimizer_config=optimizer_config, adapt_steps=adapt_steps)
    else:
        raise ValueError(f"Unsupported TTA method: {method_name}")

