# src/tta_methods/__init__.py

import torch
import torch.nn as nn
from typing import Dict, Any

from src.tta_methods.base_tta import BaseTTAMethod
from src.tta_methods.tent_tta import Tent
from src.tta_methods.cotta_tta import CoTTA

def get_tta_method(model: nn.Module, config: Dict, device: torch.device) -> BaseTTAMethod:
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
    method_name = config['tta']['method'].lower()

    if method_name == 'tent':
        # Tent constructor might need model, optimizer_config, adapt_steps
        return Tent(model=model, config=config, device=device)
    elif method_name == 'cotta':
        return CoTTA(model=model, config=config, device=device)
    else:
        raise ValueError(f"Unsupported TTA method: {method_name}")

