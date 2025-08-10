# src/tta_methods/tent_tta.py
import torch
import torch.nn as nn
from src.tta_methods.base_tta import BaseTTAMethod
from src.losses.tta_losses import EntropyMinimizationLoss
from src.utils.logger import get_logger
from typing import Dict

logger = get_logger(__name__)

class Tent(BaseTTAMethod):
    """
    Tent: Fully Test-Time Adaptation by Entropy Minimization.
    Ref: https://arxiv.org/abs/2006.10726
    This implementation is generalized to adapt either BatchNorm or LayerNorm layers.
    """
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        """
        Initializes the Tent TTA method.
        """
        super().__init__(model, device)

        tta_config = config['tta']

        self.optimizer_config = tta_config.get('optimizer', {})
        self.adaptation_params_config = tta_config.get('adaptation_params', {})
        
        self.tent_loss_fn = EntropyMinimizationLoss(task_type=config['task']['task_type']) 

        self._configure_model()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            logger.warning("Tent found no trainable parameters (BatchNorm or LayerNorm). The model will not be adapted.")
            self.optimizer = None
        else:
            self.optimizer = self._get_optimizer(trainable_params)
            logger.info(f"Initialized Tent TTA. Adapting {len(trainable_params)} parameters.")

    def _configure_model(self):
        """
        Freezes or enables trainable status of model parameters.
        Tent adapts the parameters of normalization layers (BatchNorm or LayerNorm).
        """
        self.model.eval() # Set model to eval mode globally.

        # Freeze all parameters by default.
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable gradients for normalization layers.
        for module in self.model.modules():
            # Generalized to handle both BatchNorm and LayerNorm for wider model compatibility.
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                # For BN, this updates running stats. For LN, it's just for consistency.
                module.train() 
                # Enable gradients for the affine parameters (weight & bias) of the norm layer.
                for param in module.parameters():
                    param.requires_grad = True

        trainable_params_names = [name for name, param in self.model.named_parameters() if param.requires_grad]
        logger.info(f"Tent configured. Trainable parameters: {trainable_params_names}")

    def _get_optimizer(self, params):
        """Returns an optimizer instance based on the configuration."""
        optimizer_type = self.optimizer_config.get('type', 'Adam')
        lr = self.optimizer_config.get('lr', 1e-3)
        weight_decay = self.optimizer_config.get('weight_decay', 0.0)

        if optimizer_type == 'Adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            momentum = self.optimizer_config.get('momentum', 0.9)
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def adapt(self, target_batch):
        """
        Performs a Tent adaptation step on a single batch of target data.
        """
        # If no optimizer was created (e.g., no norm layers), skip adaptation.
        if not self.optimizer:
            with torch.no_grad():
                return self.model(target_batch[0].to(self.device))

        features, _ = target_batch 
        features = features.to(self.device)

        num_steps_per_batch = self.adaptation_params_config.get('steps_per_batch', 1)
        
        for _ in range(num_steps_per_batch):
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.tent_loss_fn(logits)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            adapted_logits = self.model(features)
        
        return adapted_logits
