# src/tta_methods/adj_matrix_matcha.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from src.tta_methods.base_tta import BaseTTAMethod
from src.losses.tta_losses import PICLoss
from src.utils.logger import get_logger

logger = get_logger(__name__)

class NoOpTTA(BaseTTAMethod):
    """
    No-op TTA: forward only, no update.
    """
    def __init__(self, model: nn.Module, config: Dict, device: torch.device = None):
        super().__init__(model, config, device)
        self.optimizer = None

    def adapt(self, target_batch: tuple) -> torch.Tensor:
        self.model.eval()
        inputs, _ = target_batch
        with torch.no_grad():
            return self.model(inputs)


class AdjMatrixMatcha(BaseTTAMethod):
    """
    Matcha (PIC) adaptation for EMT: update adjs + GCN path (and tokenizers).
    Uses BaseTTA only to initialize (kept for compatibility). Pseudo labels are
    obtained by pure inference to avoid unintended parameter updates.
    """
    def __init__(self, model: nn.Module, config: Dict, device: torch.device = None):
        super().__init__(model, config, device)

        if not hasattr(self.model, 'feature_extractor') or \
           not hasattr(self.model.feature_extractor, 'adjs') or \
           not hasattr(self.model.feature_extractor, 'ge1') or \
           not hasattr(self.model.feature_extractor, 'ge2'):
            raise AttributeError("EMT feature_extractor must have 'adjs', 'ge1', and 'ge2'.")

        # 1) Initialize BaseTTA (for compatibility); do not rely on its adapt() for pseudo labels
        self.base_tta_config = self.tta_config.get('base_tta_config', {})
        self.base_tta_method_name = self.tta_config.get('base_tta_method', 'NoOpTTA')
        cfg_base = self.config.copy()
        cfg_base['tta'] = {**self.base_tta_config, 'method': self.base_tta_method_name}
        from src.tta_methods import get_tta_method
        self.base_tta_instance = get_tta_method(model=self.model, config=cfg_base, device=self.device)
        logger.info(f"AdjMatrixMatcha: BaseTTA initialized = {self.base_tta_method_name} (inference-only for pseudo labels)")

        # 2) Freeze all params, then enable gradients for target adaptation params
        for p in self.model.parameters():
            p.requires_grad_(False)

        trainable_params = []

        # adjs
        if isinstance(self.model.feature_extractor.adjs, nn.Parameter):
            self.model.feature_extractor.adjs.requires_grad_(True)
            trainable_params.append(self.model.feature_extractor.adjs)
        else:
            logger.warning("feature_extractor.adjs is not nn.Parameter; skipped.")

        # GCN layers and tokenizers in ge1/ge2
        for ge in [self.model.feature_extractor.ge1, self.model.feature_extractor.ge2]:
            if hasattr(ge, 'encoder'):
                for layer in ge.encoder:
                    if isinstance(layer, nn.Module):
                        for name, param in layer.named_parameters():
                            if 'weight' in name or 'bias' in name:
                                param.requires_grad_(True)
                                trainable_params.append(param)
            if hasattr(ge, 'tokenizer') and isinstance(ge.tokenizer, nn.Module):
                for _, param in ge.tokenizer.named_parameters():
                    param.requires_grad_(True)
                    trainable_params.append(param)

        if not trainable_params:
            raise ValueError("AdjMatrixMatcha found no trainable params (adjs/ge*/tokenizer).")

        logger.info("AdjMatrixMatcha: trainable parameters for optimizer: %d", len(trainable_params))

        # 3) Optimizer
        optimizer_type = self.tta_config.get('optimizer_type', 'Adam')
        lr = self.tta_config['lr']
        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # 4) Loss
        self.pic_loss_fn = PICLoss(temperature=self.tta_config.get('loss_temperature', 1.0))
        logger.info("PICLoss initialized with temperature: %s", self.pic_loss_fn.temperature)

    def adapt(self, target_batch: tuple) -> torch.Tensor:
        """
        One-batch PIC adaptation:
        - Generate pseudo labels from current model by pure inference (no update).
        - Run several PIC steps on EMT structure params.
        - Return final logits.
        """
        inputs, _ = target_batch
        steps = self.tta_config.get('adaptation_params', {}).get('steps_per_batch', 1)

        # Pseudo labels by pure inference (avoid calling base_tta.adapt)
        self.model.eval()
        with torch.no_grad():
            logits_ref = self.model(inputs)
            probs_ref = F.softmax(logits_ref / self.pic_loss_fn.temperature, dim=1).detach()

        # PIC adaptation steps
        for _ in range(steps):
            self.optimizer.zero_grad()
            self.model.train()
            feats = self.model.feature_extractor(inputs)  # (B, D), must keep grad
            loss = self.pic_loss_fn(feats, probs_ref)
            loss.backward()
            self.optimizer.step()

        # Final logits for evaluation
        self.model.eval()
        with torch.no_grad():
            final_logits = self.model(inputs)
        return final_logits

    def __repr__(self):
        lr_display = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') and self.optimizer.param_groups else 'N/A'
        return (f"{self.__class__.__name__}(device={self.device}, lr={lr_display}, "
                f"base_tta='{self.base_tta_method_name}', pic_temp={self.pic_loss_fn.temperature})")
