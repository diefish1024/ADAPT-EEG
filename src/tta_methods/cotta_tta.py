# src/tta_methods/cotta_tta.py

import torch
import torch.nn as nn
import copy
from typing import Dict, Tuple
from torch.nn.utils.weight_norm import WeightNorm

from src.tta_methods.base_tta import BaseTTAMethod
from src.losses.tta_losses import EntropyMinimizationLoss, ConsistencyLoss
from src.utils.logger import get_logger
from src.utils.data_augmentations import get_eeg_augmentations

logger = get_logger(__name__)

def _remove_weight_norm(model: nn.Module):
    """
    A helper function to remove weight normalization hooks from a model.
    This is necessary to enable deepcopying of models with weight_norm.
    """
    wn_modules = []
    for module in model.modules():
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                wn_modules.append(module)
                nn.utils.remove_weight_norm(module)
    return wn_modules

def _apply_weight_norm(wn_modules: list):
    """Re-applies weight normalization to a list of modules."""
    for module in wn_modules:
        nn.utils.weight_norm(module, 'weight')

class CoTTA(BaseTTAMethod):
    """
    Continual Test-Time Adaptation (CoTTA).
    Ref: https://arxiv.org/abs/2203.13591
    """
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        super().__init__(model, device)

        # Extract sub-configs from the main config dictionary
        tta_config = config.get('tta', {})
        task_config = config.get('task', {})

        self.optimizer_config = tta_config.get('optimizer', {})
        self.adaptation_params = tta_config.get('adaptation_params', {})
        
        self.restore_factor = self.adaptation_params.get('restore_factor', 0.01)
        self.aug_threshold = self.adaptation_params.get('aug_threshold', 0.92)
        
        # Safely deepcopy the model by temporarily removing weight_norm hooks
        wn_modules = _remove_weight_norm(self.model)
        self.anchor_model = copy.deepcopy(self.model).to(self.device)
        # Re-apply weight_norm to both models to restore their original state
        _apply_weight_norm(wn_modules)
        _apply_weight_norm(_remove_weight_norm(self.anchor_model))
        self.anchor_model.eval()

        self.anchor_params = {name: param.clone().detach() for name, param in self.anchor_model.named_parameters()}
        
        self.augmentations = get_eeg_augmentations(self.adaptation_params.get('augmentations', {}))

        # Correctly initialize loss functions with the required task_type
        self.loss_fn_entropy = EntropyMinimizationLoss(task_type=task_config.get('type'))
        self.loss_fn_consistency = ConsistencyLoss()

        self._configure_model()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            logger.warning("CoTTA found no trainable parameters. The model will not be adapted.")
            self.optimizer = None
        else:
            self.optimizer = self._get_optimizer(trainable_params)
            logger.info(f"Initialized CoTTA TTA. Adapting {len(trainable_params)} parameters.")
        
        logger.info(f"Restoration factor set to: {self.restore_factor}")

    def _configure_model(self):
        """Configures the model for adaptation by enabling gradients for norm layers."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                module.train()
                for param in module.parameters():
                    param.requires_grad = True
        
        trainable_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        logger.info(f"CoTTA configured. Trainable parameters: {trainable_names}")

    def _get_optimizer(self, params):
        """Returns an optimizer instance based on the configuration."""
        optimizer_type = self.optimizer_config.get('type', 'Adam')
        lr = self.optimizer_config.get('lr', 1e-3)
        return torch.optim.Adam(params, lr=lr)

    def adapt(self, target_batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Performs a complete CoTTA adaptation step."""
        features, _ = target_batch
        
        if not self.optimizer:
            with torch.no_grad():
                return self.model(features)

        logits = self.model(features)
        loss = self.loss_fn_entropy(logits)
        
        with torch.no_grad():
            anchor_logits = self.anchor_model(features)
        
        if torch.max(torch.softmax(anchor_logits, dim=1)) < self.aug_threshold:
            augmented_features = self.augmentations(features)
            augmented_logits = self.model(augmented_features)
            consistency_loss = self.loss_fn_consistency(logits, augmented_logits.detach())
            loss += consistency_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    anchor_param = self.anchor_params[name]
                    param.data = self.restore_factor * anchor_param.data + (1 - self.restore_factor) * param.data
        
        with torch.no_grad():
            adapted_logits = self.model(features)
        
        return adapted_logits
