# src/losses/__init__.py

import torch.nn as nn
from typing import Dict, Any
import logging

from .classification_losses import CrossEntropyLoss
from .regression_losses import NLLLoss, MSELoss
from .tta_losses import (
    EntropyMinimizationLoss,
    UncertaintyWeightedConsistencyLoss,
    ConsistencyLoss,
    InfoNCELoss,
    DomainAdversarialLoss,
    WeightedCORALLoss,
    UncertaintyWeightedPseudoLabelLoss
)

logger = logging.getLogger(__name__)

def get_losses(config: Dict[str, Any]) -> Dict[str, nn.Module]:
    """
    Instantiates and returns a dictionary of loss functions based on the configuration.
    
    Args:
        config (Dict[str, Any]): The full experiment configuration dictionary.
        
    Returns:
        Dict[str, nn.Module]: A dictionary where keys are descriptive names for loss functions
                              (e.g., 'source_loss', 'tta_entropy_loss_fn') and values are
                              their instantiated nn.Module objects.
    """
    loss_fns: Dict[str, nn.Module] = {}

    # --- 1. Instantiate Source Loss ---
    task_type = config['task']['type'].lower()
    source_loss_name = config['task']['loss_function']

    if task_type == 'classification':
        if source_loss_name == 'CrossEntropyLoss':
            loss_fns['source_loss'] = CrossEntropyLoss()
            logger.info("Instantiated source loss: CrossEntropyLoss.")
        else:
            raise ValueError(f"Unsupported classification source loss function: {source_loss_name}")
    elif task_type == 'regression':
        if source_loss_name == 'NLLLoss': # For uncertainty regression (primarily DEAP)
            loss_fns['source_loss'] = NLLLoss()
            logger.info("Instantiated source loss: NLLLoss.")
        elif source_loss_name == 'MSELoss': # For general regression
            loss_fns['source_loss'] = MSELoss()
            logger.info("Instantiated source loss: MSELoss.")
        else:
            raise ValueError(f"Unsupported regression source loss function: {source_loss_name}")
    else:
        raise ValueError(f"Unsupported task type for source loss instantiation: {task_type}")

    # --- 2. Instantiate TTA Losses (Conditional based on TTA config) ---
    if config['tta']['enable']:
        tta_method = config['tta']['method'].lower()
        tta_loss_config = config['tta'].get('losses', {}) 

        # Common TTA Loss: Entropy Minimization
        if tta_loss_config.get('entropy_minimization', {}).get('enable', False):
            loss_fns['tta_entropy_loss_fn'] = EntropyMinimizationLoss(task_type=task_type)
            logger.info(f"Instantiated TTA loss: EntropyMinimizationLoss (for task type: {task_type}).")

        # Consistency Loss (e.g., for CoTTA)
        if tta_loss_config.get('consistency', {}).get('enable', False):
            loss_fns['tta_consistency_loss_fn'] = ConsistencyLoss()
            logger.info("Instantiated TTA loss: ConsistencyLoss.")

        if task_type == 'regression':
            if tta_loss_config.get('uncertainty_weighted_consistency', {}).get('enable', False):
                loss_fns['tta_uncertainty_weighted_consistency_loss_fn'] = UncertaintyWeightedConsistencyLoss()
                logger.info("Instantiated TTA loss: UncertaintyWeightedConsistencyLoss.")
            
            if tta_loss_config.get('uncertainty_weighted_pseudo_label', {}).get('enable', False):
                loss_fns['tta_uncertainty_weighted_pseudo_label_loss_fn'] = UncertaintyWeightedPseudoLabelLoss()
                logger.info("Instantiated TTA loss: UncertaintyWeightedPseudoLabelLoss.")
            
            if tta_loss_config.get('info_nce', {}).get('enable', False):
                temperature = tta_loss_config['info_nce'].get('temperature', 0.07) # Default temperature
                loss_fns['tta_info_nce_loss_fn'] = InfoNCELoss(temperature=temperature)
                logger.info(f"Instantiated TTA loss: InfoNCELoss (temperature={temperature}).")
            
            if tta_loss_config.get('domain_adversarial', {}).get('enable', False):
                loss_fns['tta_domain_adversarial_loss_fn'] = DomainAdversarialLoss()
                logger.info("Instantiated TTA loss: DomainAdversarialLoss.")
            
            if tta_loss_config.get('weighted_coral', {}).get('enable', False):
                loss_fns['tta_weighted_coral_loss_fn'] = WeightedCORALLoss()
                logger.info("Instantiated TTA loss: WeightedCORALLoss.")
        
    else:
        logger.info("TTA is disabled in config. No TTA-specific losses instantiated.")

    return loss_fns

