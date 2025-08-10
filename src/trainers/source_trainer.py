# src/trainers/source_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
from tqdm import tqdm
import os
import copy
from typing import Dict, Union, Optional
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.metrics import get_metrics_calculator
from src.utils.early_stopping import EarlyStopping

# Import specific loss functions
from src.losses.classification_losses import CrossEntropyLoss  # Assuming this exists and is standard nn.CrossEntropyLoss
from src.losses.regression_losses import NLLLoss # Assuming NLLLoss is defined here as discussed

logger = get_logger(__name__)

class SourceTrainer:
    """
    Trainer class for pre-training the model on the source domain.
    Handles the standard training loop including optimization, validation,
    checkpointing, and early stopping.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 config: Dict, checkpoint_dir: Path, device: torch.device):
        """
        Initializes the Source Trainer.

        Args:
            model (nn.Module): The model to be trained.
            train_loader (DataLoader): DataLoader for the source domain training data.
            val_loader (DataLoader): DataLoader for the source domain validation data.
            optimizer_config (Dict): Configuration for the optimizer (e.g., type, learning rate).
            config (Dict): The overall experiment configuration dictionary.
            device (torch.device): The device (CPU/GPU) to run computations on.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.task_type = config['task']['type']
        self.loss_fn = self._get_loss_fn()
        self.optimizer = self._get_optimizer(self.model.parameters(), config['training']['optimizer'])
        self.metrics_calculator = get_metrics_calculator(self.task_type)
        
        self.epochs = config['training']['epochs']
        
        # Setup early stopping
        early_stopping_config = config['training']['early_stopping']
        early_stopping_monitor_metric = early_stopping_config['monitor_metric']
        early_stopping_patience = early_stopping_config['patience']
        early_stopping_mode = early_stopping_config['mode']
        self.early_stopping = EarlyStopping(
            monitor=early_stopping_monitor_metric,
            patience=early_stopping_patience,
            mode=early_stopping_mode,
            verbose=True
        )

        clipping_config = self.config['training'].get('gradient_clipping', {})
        self.clip_grad_enabled = clipping_config.get('enable', False)
        self.clip_max_norm = clipping_config.get('max_norm', 1.0)
        if self.clip_grad_enabled:
            logger.info(f"Gradient clipping enabled with max_norm={self.clip_max_norm}.")
        
        # Setup directories for saving results and checkpoints
        self.checkpoint_dir = checkpoint_dir
        
        self.best_val_metric = -float('inf') if early_stopping_mode == 'max' else float('inf')
        self.best_epoch = 0
        self.best_model_state = None

        logger.info(f"SourceTrainer initialized for task: {self.task_type} on device: {self.device}")
        logger.info(f"Model will be trained for {self.epochs} epochs.")
        logger.info(f"Loss function: {self.loss_fn.__class__.__name__}, Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Early stopping monitoring '{early_stopping_monitor_metric}' with patience {early_stopping_patience} in '{early_stopping_mode}' mode.")

    def _get_loss_fn(self) -> nn.Module:
        """
        Selects and returns the appropriate loss function based on the task type.
        """
        if self.task_type == 'classification':
            # CrossEntropyLoss expects raw logits and integer labels
            return CrossEntropyLoss() # Assuming CrossEntropyLoss is just nn.CrossEntropyLoss
        elif self.task_type == 'regression':
            # NLLLoss expects (y_true, mu_pred, log_sigma_sq_pred)
            return NLLLoss()
        else:
            raise ValueError(f"Unsupported task type for loss function: {self.task_type}")

    def _get_optimizer(self, params, optimizer_config: Dict) -> torch.optim.Optimizer:
        """
        Constructs and returns the optimizer based on configuration.
        """
        opt_type = optimizer_config['type'].lower()
        lr = optimizer_config['lr']
        weight_decay = optimizer_config.get('weight_decay', 0)
        if opt_type == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_type == 'sgd':
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=optimizer_config.get('momentum', 0))
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

    def _validate(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Performs validation step on the validation dataset.

        Returns:
            Dict[str, Union[float, Dict[str, float]]]: Dictionary of validation metrics.
        """
        self.model.eval() # Set model to evaluation mode
        
        all_model_outputs = [] # Stores raw model outputs for metrics (logits or mu_pred)
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad(): # Disable gradient calculations for validation
            for features, labels in self.val_loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(features) # Will be logits or (mu_pred, log_sigma_sq_pred)

                if self.task_type == 'classification':
                    loss = self.loss_fn(outputs, labels)
                    all_model_outputs.append(outputs) # Append logits
                elif self.task_type == 'regression':
                    mu_pred, log_sigma_sq_pred = outputs # Unpack tuple for regression head output
                    loss = self.loss_fn(labels, mu_pred, log_sigma_sq_pred)
                    all_model_outputs.append(mu_pred) # Append only mu_pred for standard regression metrics
                
                all_targets.append(labels)
                total_loss += loss.item() * features.size(0) # Accumulate sum of losses
        
        # Concatenate all collected tensors
        final_targets = torch.cat(all_targets, dim=0) if all_targets else torch.tensor([])
        final_model_outputs = torch.cat(all_model_outputs, dim=0) if all_model_outputs else torch.tensor([])

        avg_loss = total_loss / len(self.val_loader.dataset) if len(self.val_loader.dataset) > 0 else 0.0
        metrics = self.metrics_calculator(final_targets, final_model_outputs)
        metrics['loss'] = avg_loss
        
        self.model.train() # Set model back to training mode
        return metrics

    def _save_checkpoint(self, epoch: int, metric_value: float, is_best: bool = False) -> None:
        """
        Saves the model and optimizer state.

        Args:
            epoch (int): The current epoch number.
            metric_value (float): The value of the monitored metric.
            is_best (bool): True if this is the best model so far.
        """
        checkpoint_name = f"epoch_{epoch:03d}_{self.early_stopping.monitor}_{metric_value:.4f}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config, # Optionally save full config
        }
        
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_model_path)
            logger.info(f"New best model saved to {best_model_path}")

    def train(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Executes the main training loop for the source domain.

        Returns:
            Dict[str, Union[float, Dict[str, float]]]: Metrics of the best model found during training.
        """
        logger.info("Starting source domain pre-training.")
        logger.info(f"Training DataLoader will have {len(self.train_loader)} batches per epoch.")
        
        for epoch in range(self.epochs):
            self.model.train() # Ensure model is in training mode at the start of each epoch
            total_train_loss = 0.0
            
            # Use tqdm for a progress bar during training
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]", leave=False)
            for batch_idx, (features, labels) in enumerate(train_pbar):
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad() # Zero gradients before each batch
                outputs = self.model(features) # Forward pass

                if self.task_type == 'classification':
                    loss = self.loss_fn(outputs, labels)
                elif self.task_type == 'regression':
                    mu_pred, log_sigma_sq_pred = outputs # Unpack tuple
                    loss = self.loss_fn(labels, mu_pred, log_sigma_sq_pred)
                
                loss.backward() # Backward pass
                if self.clip_grad_enabled:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
                self.optimizer.step() # Update model parameters

                total_train_loss += loss.item() * features.size(0)
                train_pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = total_train_loss / len(self.train_loader.dataset)
            logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

            # Validation step
            val_metrics = self._validate()
            val_loss = val_metrics['loss']
            
            # Log validation metrics
            logger.info(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    logger.info(f"Val {k}: {v:.4f}")
                elif isinstance(v, dict):
                    logger.info(f"Val {k}: {v}")

            # Check for early stopping
            monitor_value = val_metrics[self.early_stopping.monitor]
            self.early_stopping(monitor_value, self.model)
            
            # Update best model and save checkpoint
            if self.early_stopping.is_best_epoch:
                self.best_val_metric = monitor_value
                self.best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(epoch + 1, monitor_value, is_best=True)
            else:
                self._save_checkpoint(epoch + 1, monitor_value, is_best=False) # Always save epoch checkpoint
                
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}. No improvement for {self.early_stopping.patience} epochs.")
                break

        logger.info(f"Training finished. Best model from Epoch {self.best_epoch} with {self.early_stopping.monitor}: {self.best_val_metric:.4f}")
        
        # Load the best model state dictionary before returning
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state for final return.")
            
        # Perform one final evaluation of the best model (if applicable)
        final_best_model_metrics = self._validate()
        return final_best_model_metrics

