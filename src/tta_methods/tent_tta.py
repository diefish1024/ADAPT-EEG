# src/tta_methods/tent_tta.py
import torch
import torch.nn as nn
from src.tta_methods.base_tta import BaseTTAMethod
from src.losses.tta_losses import EntropyMinimizationLoss
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Tent(BaseTTAMethod):
    """
    Tent: Entropy Minimization for Test-Time Adaptation.
    Ref: https://arxiv.org/abs/2006.00295
    """
    def __init__(self, model, optimizer_config, adaptation_params_config, **kwargs):
        """
        Initializes the Tent TTA method.

        Args:
            model (nn.Module): The pre-trained model.
            optimizer_config (dict): Optimizer configuration for the TTA phase.
            adaptation_params_config (dict): Adaptation parameters configuration, e.g., whether to adapt only BN layers.
            **kwargs: Other parameters passed to the base class.
        """
        super().__init__(model, **kwargs)

        # Tent typically operates with the model primarily in evaluation mode,
        # but specifically updates BN running statistics and affine parameters.
        self.model.eval()  
        self.optimizer_config = optimizer_config
        self.adaptation_params_config = adaptation_params_config
        
        # Initialize the entropy minimization loss for classification tasks (e.g., on SEED).
        self.tent_loss_fn = EntropyMinimizationLoss(task_type='classification') 

        # Configure model parameters for adaptation.
        self._configure_model()

        # Initialize the optimizer, optimizing only the trainable parameters.
        self.optimizer = self._get_optimizer([p for p in self.model.parameters() if p.requires_grad])
        logger.info(f"Initialized Tent TTA. Adapting {len([p for p in self.model.parameters() if p.requires_grad])} parameters.")

    def _configure_model(self):
        """
        Freezes or enables trainable status of model parameters based on configuration.
        By default, Tent adapts only the affine parameters and running statistics of Batch Normalization layers.
        """
        # Iterate through all parameters, freezing them by default.
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable trainable parameters for BN layers.
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Enables gradients for affine parameters (weight, bias).
                module.requires_grad_(True) 
                
                # The original Tent paper updates running_mean/running_variance.
                # PyTorch's BN layers do not update running stats in model.eval() mode.
                # Therefore, we set BN layers to train mode here to ensure they are updated.
                # However, be aware that this might cause other layers to exhibit training mode behavior
                # if not carefully handled.
                module.train() # Ensures running stats are updated, while affine parameters are also updated.

        # Set the entire model to evaluation mode, which will affect layers other than BN.
        self.model.eval() 

        # Logging which parameters are trainable.
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
        Performs a Tent adaptation step.

        Args:
            target_batch (tuple): A batch containing target domain data, typically (features, [labels]).
                                  For Tent, labels are optional as its TTA is self-supervised.
        Returns:
            torch.Tensor: The model's predictions (logits) after adaptation.
        """
        # For TTA, we typically do not use target domain labels.
        features, _ = target_batch 
        features = features.to(self.device)

        # Perform multiple adaptation steps per batch for better convergence.
        num_steps_per_batch = self.adaptation_params_config.get('steps_per_batch', 1)
        
        for step in range(num_steps_per_batch):
            self.optimizer.zero_grad() # Zero the gradients.

            # Forward pass to get model output logits.
            logits = self.model(features)
            
            # Compute Tent loss (entropy minimization).
            loss = self.tent_loss_fn(logits)

            # Backpropagation and optimization.
            loss.backward()
            self.optimizer.step()

        # Return the model's predictions after adaptation on the current batch (for evaluation).
        # The model has been updated; perform another forward pass to get the latest predictions.
        with torch.no_grad(): # No need to compute gradients when returning predictions.
            adapted_logits = self.model(features)
        
        return adapted_logits
