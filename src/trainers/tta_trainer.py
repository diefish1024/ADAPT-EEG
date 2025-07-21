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

        Args:
            model (nn.Module): The pre-trained model to be adapted.
            test_loader (DataLoader): DataLoader for the target domain test data.
            config (Dict): The overall experiment configuration dictionary.
                           Must contain 'tta' and 'task' keys.
            device (torch.device): The device (CPU/GPU) to run computations on.
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.task_type = config['task']['type']
        self.result_dir = results_dir
        
        # Instantiate the TTA method internally using the config. This unifies with SourceTrainer's approach.
        self.tta_method = get_tta_method(
            model=self.model,           # Pass the model to the TTA method for adaptation
            tta_config=config['tta'],   # Pass the TTA specific part of the config
            device=self.device
        )
        
        # Initialize the appropriate metrics calculator based on task type
        self.metrics_calculator = get_metrics_calculator(self.task_type)
        
        logger.info(f"TTATrainer initialized for task: {self.task_type} using TTA method: {self.tta_method.__class__.__name__} on device: {device}")

    def evaluate(self, data_loader: Optional[DataLoader] = None, model_to_eval: Optional[nn.Module] = None) -> Dict[str, Union[float, Dict[str, float]]]:
        if data_loader is None:
            data_loader = self.test_loader
        if model_to_eval is None:
            model_to_eval = self.model

        model_to_eval.eval() # Set model to evaluation mode (important for BN and Dropout layers)
        
        all_model_outputs = [] # To store raw model outputs (logits for classification, mu_pred for regression)
        all_targets = []
        total_loss = 0.0
        
        # Determine the appropriate evaluation loss function based on task type
        eval_loss_fn: nn.Module
        if self.task_type == 'classification':
            eval_loss_fn = CrossEntropyLoss(reduction='sum') # Use sum reduction to accumulate total loss
        elif self.task_type == 'regression':
            eval_loss_fn = NLLLoss(reduction='sum') # Use NLL for consistency with regression head (if it outputs uncertainty)
        else:
            raise ValueError(f"Unknown task type for evaluation loss: {self.task_type}")

        # Disable gradient calculations for evaluation to save memory and computation
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(data_loader):
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = model_to_eval(features) # Will be logits or (mu_pred, log_sigma_sq_pred)

                if self.task_type == 'classification':
                    loss = eval_loss_fn(outputs, labels)
                    all_model_outputs.append(outputs) # Store logits
                elif self.task_type == 'regression':
                    # Assuming uncertainty_regression_head outputs a tuple (mu_pred, log_sigma_sq_pred)
                    mu_pred, log_sigma_sq_pred = outputs
                    loss = eval_loss_fn(labels, mu_pred, log_sigma_sq_pred)
                    all_model_outputs.append(mu_pred) # Store only mu_pred for standard regression metrics (MSE, R2)

                total_loss += loss.item()
                all_targets.append(labels)

        # Concatenate all collected tensors for batch evaluation
        final_targets = torch.cat(all_targets, dim=0)
        final_model_outputs = torch.cat(all_model_outputs, dim=0)

        # Calculate average loss over the entire dataset
        avg_loss = total_loss / len(data_loader.dataset)
        
        # Calculate performance metrics using the centralized metrics calculator
        metrics = self.metrics_calculator(final_targets, final_model_outputs)
        metrics['loss'] = avg_loss
        
        logger.info(f"Evaluation completed. Loss: {avg_loss:.4f}")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                logger.info(f"- {k}: {v:.4f}")
            elif isinstance(v, dict): # For multi-dimensional metrics like r2_per_dim
                logger.info(f"- {k}: {v}") # Log the dictionary as is
        return metrics

    def adapt_and_evaluate(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Executes the Test-Time Adaptation (TTA) process on the model
        this trainer was initialized with.

        Returns:
            dict: Final evaluation metrics after the TTA process is complete.
        """
        logger.info(f"Starting TTA process using method: {self.tta_method.__class__.__name__}")
        
        # 1. The model is already loaded by main.py. No need to load it here.
        #    We proceed directly to the initial evaluation.
        logger.info("--- Initial Evaluation (Before TTA) ---")
        initial_metrics = self.evaluate(data_loader=self.test_loader, model_to_eval=self.model)
        
        # 2. Set model to train mode for TTA. The TTA method instance will internally manage
        #    which parameters are trainable.
        self.model.train() 

        logger.info(f"Performing TTA adaptation on {len(self.test_loader)} target domain batches.")
        # 3. Perform adaptation on each batch of the target domain data.
        for batch_idx, (features, labels) in enumerate(tqdm(self.test_loader, desc="Adapting on Target Domain Batches")):
            # The `adapt` method modifies self.model parameters in-place.
            self.tta_method.adapt((features, labels))
            
        # 4. Final evaluation after all batches have been used for adaptation.
        logger.info("--- Final Evaluation (After TTA on all batches) ---")
        self.model.eval() # Ensure model is in eval mode for final assessment
        final_metrics = self.evaluate(data_loader=self.test_loader, model_to_eval=self.model)

        # 5. Log summary of results.
        logger.info(
            f"TTA run completed. "
            f"Initial Accuracy: {initial_metrics.get('accuracy', 'N/A'):.4f} "
            f"-> Final Accuracy: {final_metrics.get('accuracy', 'N/A'):.4f}"
        )
        return final_metrics

