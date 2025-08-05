# src/trainers/tta_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.metrics import get_metrics_calculator
from src.utils.logger import get_logger
import os
from typing import Dict, Union, Optional
from pathlib import Path

from src.losses.classification_losses import CrossEntropyLoss
from src.losses.regression_losses import NLLLoss 
from src.tta_methods import get_tta_method

logger = get_logger(__name__)

class TTATrainer:
    """
    Trainer class for Test-Time Adaptation (TTA) experiments.
    Manages loading pre-trained models, running the TTA adaptation loop,
    and evaluating the model's performance on the target domain.
    """
    def __init__(self, model: nn.Module, test_loader: DataLoader, config: Dict,
                 results_dir: Path, device: torch.device):
        """
        Initializes the TTA Trainer.
        """
        if 'tta' not in config or 'task' not in config:
            raise KeyError("The 'config' dictionary must contain 'tta' and 'task' keys.")

        self.model = model.to(device)
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.task_type = config['task']['type']
        self.result_dir = results_dir
        
        self.tta_method = get_tta_method(
            model=self.model,
            tta_config=config['tta'],
            device=device
        )
        
        self.metrics_calculator = get_metrics_calculator(self.task_type)
        
        logger.info(f"TTATrainer initialized for task: {self.task_type} using TTA method: {self.tta_method.__class__.__name__} on device: {device}")

    def evaluate(self, data_loader: Optional[DataLoader] = None, model_to_eval: Optional[nn.Module] = None) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Performs a standard evaluation run without adaptation.
        """
        if data_loader is None:
            data_loader = self.test_loader
        if model_to_eval is None:
            model_to_eval = self.model

        model_to_eval.eval()
        
        all_model_outputs = []
        all_targets = []
        total_loss = 0.0
        
        eval_loss_fn: nn.Module
        if self.task_type == 'classification':
            eval_loss_fn = CrossEntropyLoss(reduction='sum')
        elif self.task_type == 'regression':
            eval_loss_fn = NLLLoss(reduction='sum')
        else:
            raise ValueError(f"Unknown task type for evaluation loss: {self.task_type}")

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = model_to_eval(features)

                if self.task_type == 'classification':
                    loss = eval_loss_fn(outputs, labels)
                    all_model_outputs.append(outputs)
                elif self.task_type == 'regression':
                    mu_pred, log_sigma_sq_pred = outputs
                    loss = eval_loss_fn(labels, mu_pred, log_sigma_sq_pred)
                    all_model_outputs.append(mu_pred)

                total_loss += loss.item()
                all_targets.append(labels)

        final_targets = torch.cat(all_targets, dim=0)
        final_model_outputs = torch.cat(all_model_outputs, dim=0)
        avg_loss = total_loss / len(data_loader.dataset)
        
        metrics = self.metrics_calculator(final_targets, final_model_outputs)
        metrics['loss'] = avg_loss
        
        logger.info(f"Evaluation completed. Loss: {avg_loss:.4f}")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                logger.info(f"- {k}: {v:.4f}")
            elif isinstance(v, dict):
                logger.info(f"- {k}: {v}")
        return metrics

    def adapt_and_evaluate(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Executes an optimized Test-Time Adaptation (TTA) and evaluation process.
        This version adapts and evaluates in a single pass over the data.
        """
        logger.info(f"Starting TTA process using method: {self.tta_method.__class__.__name__}")
        
        logger.info("--- Initial Evaluation (Before TTA) ---")
        initial_metrics = self.evaluate(data_loader=self.test_loader, model_to_eval=self.model)
        
        logger.info(f"--- Online Adaptation and Evaluation on {len(self.test_loader)} target batches ---")
        
        all_adapted_outputs = []
        all_targets = []

        # The TTA method itself sets the appropriate train/eval modes for layers.
        for features, labels in tqdm(self.test_loader, desc="Adapting and Evaluating"):
            target_batch = (features.to(self.device), labels.to(self.device))
            
            # The `adapt` method performs in-place updates and returns the adapted logits for the current batch.
            adapted_outputs = self.tta_method.adapt(target_batch)
            
            # Collect results for final metric calculation.
            if self.task_type == 'classification':
                all_adapted_outputs.append(adapted_outputs)
            elif self.task_type == 'regression':
                all_adapted_outputs.append(adapted_outputs[0]) # Store only mu_pred

            all_targets.append(target_batch[1])

        # After the loop, calculate metrics on all the collected adapted outputs.
        # This avoids a second, redundant pass over the test_loader.
        final_targets = torch.cat(all_targets, dim=0)
        final_model_outputs = torch.cat(all_adapted_outputs, dim=0)
        
        # We calculate metrics directly, loss calculation is omitted here as it's less meaningful
        # in an online setting where the model is constantly changing.
        final_metrics = self.metrics_calculator(final_targets, final_model_outputs)
        
        logger.info("--- Final Evaluation (After TTA on all batches) ---")
        for k, v in final_metrics.items():
            if isinstance(v, (int, float)):
                logger.info(f"- {k}: {v:.4f}")
            elif isinstance(v, dict):
                logger.info(f"- {k}: {v}")

        logger.info(
            f"TTA run completed. "
            f"Initial Accuracy: {initial_metrics.get('accuracy', 'N/A'):.4f} "
            f"-> Final Accuracy: {final_metrics.get('accuracy', 'N/A'):.4f}"
        )
        return final_metrics