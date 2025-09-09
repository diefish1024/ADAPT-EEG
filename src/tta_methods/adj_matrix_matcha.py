# src/tta_methods/adj_matrix_matcha.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from src.tta_methods.base_tta import BaseTTAMethod
from src.losses.tta_losses import PICLoss
from src.utils.logger import get_logger
from typing import Dict

logger = get_logger(__name__)

class NoOpTTA(BaseTTAMethod):
    """
    A 'No-Operation' TTA method used when no specific adaptation is desired.
    It simply performs a forward pass without any model adaptation.
    """

    def __init__(self, model: nn.Module, config: Dict, device: torch.device = None):
        super().__init__(model, config, device)
        self.optimizer = None
        # logger.debug("NoOpTTA initialized.")

    def adapt(self, target_batch: tuple) -> torch.Tensor:
        """
        Performs a forward pass without adaptation.

        Args:
            target_batch (tuple): A batch containing target domain features (inputs) and labels.

        Returns:
            torch.Tensor: The model's output (logits).
        """
        self.model.eval()  # Set model to evaluation mode for inference
        inputs, _ = target_batch
        with torch.no_grad():
            logits = self.model(inputs)
        return logits


class AdjMatrixMatcha(BaseTTAMethod):
    """
    AdjMatrixMatcha for Test-Time Adaptation of EMT model's learnable adjacency matrices.
    It combines Matcha's PIC Loss with an optional BaseTTA method to
    adapt the graph structure parameters (self.adjs and GCN weights) of the EMT model.
    """

    def __init__(self, model: nn.Module, config: Dict, device: torch.device = None):
        super().__init__(model, config, device)
        # logger.debug(f"AdjMatrixMatcha Config: {self.tta_config}")

        if not hasattr(self.model, 'feature_extractor') or \
           not hasattr(self.model.feature_extractor, 'adjs') or \
           not hasattr(self.model.feature_extractor, 'ge1') or \
           not hasattr(self.model.feature_extractor, 'ge2'):
            raise AttributeError("Model must be a BaseModel with EMTFeatureExtractor "
                                 "containing 'adjs', 'ge1', and 'ge2' attributes for AdjMatrixMatcha.")

        # --- Configure parameters to optimize ---
        # Core strategy: Freeze all parameters by default, then explicitly enable requires_grad for adaptive parameters.
        # This avoids frequent resetting of requires_grad during each adapt() call, reducing overhead and ensuring clear logic.
        for param in self.model.parameters():
            param.requires_grad_(False)  # Freeze all model parameters by default

        # for name, param in self.model.named_parameters():
        #     logger.debug(f"Param: {name}, Shape: {param.shape}, Is_leaf: {param.is_leaf}, Requires_grad: {param.requires_grad}")

        trainable_params = []

        # 1. Learnable adjacency matrices (self.adjs)
        if isinstance(self.model.feature_extractor.adjs, nn.Parameter):
            self.model.feature_extractor.adjs.requires_grad_(True)  # Enable gradients
            trainable_params.append(self.model.feature_extractor.adjs)
            # logger.debug("Adjacency matrices (adjs) will be adapted.")
        else:
            logger.warning("Model's feature_extractor.adjs is not an nn.Parameter; it will not be adapted.")

        # 2. GCN layer weights and biases within GraphEncoders (ge1, ge2)
        for ge_idx, ge in enumerate([self.model.feature_extractor.ge1, self.model.feature_extractor.ge2]):
            for layer_idx, layer in enumerate(ge.encoder):
                if isinstance(layer, nn.Module):
                    for name, param in layer.named_parameters():
                        # Adapt only weights and biases; freeze others (if any)
                        if 'weight' in name or 'bias' in name:
                            param.requires_grad_(True)  # Enable gradients
                            trainable_params.append(param)
                            # logger.debug(f"GCN layer parameter '{name}' in {ge.__class__.__name__} (GE{ge_idx+1}-Layer{layer_idx+1}, shape={param.shape}) will be adapted.")
            
            # Tokenizer parameters
            if hasattr(ge, 'tokenizer') and isinstance(ge.tokenizer, nn.Module):
                for name, param in ge.tokenizer.named_parameters():
                    param.requires_grad_(True)  # Enable gradients
                    trainable_params.append(param)
                    # logger.debug(f"GraphEncoder tokenizer parameter '{name}' in {ge.__class__.__name__} (GE{ge_idx+1}, shape={param.shape}) will be adapted.")
        
        # 3. Classifier head parameters (restoring adaptation for the classifier head, common in Matcha)
        for name, param in self.model.head.named_parameters():
            param.requires_grad_(True)  # Enable gradients
            trainable_params.append(param)
            # logger.debug(f"Classifier head parameter '{name}' (shape={param.shape}) will be adapted.")

        # Filter parameters to ensure only those explicitly marked for gradient computation are included for the optimizer.
        final_trainable_params_filtered = []
        for p in trainable_params:
            if p.requires_grad:
                final_trainable_params_filtered.append(p)
                # logger.debug("AdjMatrixMatcha DEBUG: Kept parameter for optimization (shape=%s, is_leaf=%s, requires_grad=%s).", 
                #              p.shape, p.is_leaf, p.requires_grad)
            # else:
            #     logger.debug("AdjMatrixMatcha DEBUG: Skipped parameter (shape=%s, is_leaf=%s, requires_grad=%s) as not requiring grad.",
            #                  getattr(p, 'shape', 'N/A'), getattr(p, 'is_leaf', 'N/A'), getattr(p, 'requires_grad', 'N/A'))
        
        trainable_params_for_optimizer = final_trainable_params_filtered

        logger.info("AdjMatrixMatcha: Final count of trainable parameters for optimizer: %d.", len(trainable_params_for_optimizer))
        if not trainable_params_for_optimizer:
            raise ValueError("No trainable parameters found for AdjMatrixMatcha. "
                             "Ensure at least one of (adjs, GCN layers, classifier head) exists and is set to require_grad=True.")

        # --- Initialize Optimizer ---
        optimizer_type = self.tta_config.get('optimizer_type', 'Adam')
        lr = self.tta_config.get('lr', 0.001)
        # logger.debug("AdjMatrixMatcha: Initializing optimizer '%s' with lr=%f for %d parameters.",
        #              optimizer_type, lr, len(trainable_params_for_optimizer))

        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(trainable_params_for_optimizer, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # --- Initialize Loss Function ---
        self.pic_loss_fn = PICLoss(temperature=self.tta_config.get('loss_temperature', 1.0))
        logger.info(f"PICLoss initialized with temperature: {self.pic_loss_fn.temperature}")

        # --- Initialize BaseTTA ---
        self.base_tta_config = self.tta_config.get('base_tta_config', {})
        self.base_tta_method_name = self.tta_config.get('base_tta_method', 'NoOpTTA')
        config_for_get_tta = self.config.copy()
        # Override the 'tta' key in the copied configuration with the specific settings for the base TTA
        config_for_get_tta['tta'] = {
            **self.base_tta_config,
            'method': self.base_tta_method_name
        }
        from src.tta_methods import get_tta_method
        self.base_tta_instance = get_tta_method(
            model=model,
            config=config_for_get_tta,
            device=self.device
        )

        logger.info(f"AdjMatrixMatcha initialized. Using BaseTTA method for pseudo-labeling: {self.base_tta_method_name}")


    def adapt(self, target_batch: tuple) -> torch.Tensor:
        """
        Adapts the model's learnable adjacency matrices and GCN & Classifier parameters
        using PIC Loss, guided by pseudo-labels from a BaseTTA method.

        Args:
            target_batch (tuple): A batch containing target domain features (inputs) and labels.

        Returns:
            torch.Tensor: The model's output (logits) after adaptation on the current batch.
        """
        inputs, labels = target_batch
        adapt_steps_per_batch = self.tta_config.get('adaptation_params', {}).get('steps_per_batch', 1)

        tta_logits_for_pseudo_labels = self.base_tta_instance.adapt(target_batch)
        prob_batch_level = F.softmax(tta_logits_for_pseudo_labels / self.pic_loss_fn.temperature, dim=1).detach()

        for step in range(adapt_steps_per_batch):
            self.optimizer.zero_grad()
            self.model.train()
            current_feats = self.model.feature_extractor(inputs)
            loss = self.pic_loss_fn(current_feats, prob_batch_level)
            loss.backward()
            self.optimizer.step()

        # After adaptation, switch the model back to eval mode.
        # Batch Norm's running_mean/running_var will now reflect the adapted state.
        self.model.eval() 
        # Perform final forward pass in eval mode and without gradient tracking.
        with torch.no_grad():
            final_logits = self.model(inputs)
        
        return final_logits

    def __repr__(self):
        lr_display = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') and self.optimizer.param_groups else 'N/A'
        return (f"{self.__class__.__name__}(device={self.device}, lr={lr_display},"
                f" base_tta='{self.base_tta_method_name}', pic_temp={self.pic_loss_fn.temperature})")
