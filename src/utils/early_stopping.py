# src/utils/early_stopping.py
import numpy as np
import torch
from typing import Literal
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, monitor: str = 'loss', patience: int = 7, verbose: bool = False, 
                 delta: float = 0, path: str = 'checkpoint.pt', 
                 mode: Literal['min', 'max'] = 'min'):
        """
        Args:
            monitor (str): Name of the metric to monitor (e.g., 'loss', 'accuracy', 'r2').
            patience (int): How many epochs to wait after last best validation result.
                            Default: 7
            verbose (bool): If True, prints a message for each validation metric improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the 
                        quantity monitored has stopped decreasing; in 'max' mode it will 
                        stop when the quantity monitored has stopped increasing. Default: 'min'
        """
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_best_epoch = False # New flag to indicate if current epoch is the best one
        
        if self.mode == 'min':
            self.val_score_is_better = lambda current, best: current < best - self.delta
            self.best_score = np.Inf
        elif self.mode == 'max':
            self.val_score_is_better = lambda current, best: current > best + self.delta
            self.best_score = -np.Inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', but got {mode}")

    def __call__(self, val_metric: float, model: torch.nn.Module) -> None:
        """
        Checks if training should stop based on the monitored validation metric.

        Args:
            val_metric (float): The current value of the monitored metric.
            model (torch.nn.Module): The model instance (not used for saving here, trainer handles it).
        """
        self.is_best_epoch = False
        if self.val_score_is_better(val_metric, self.best_score):
            if self.verbose:
                if self.mode == 'min':
                    logger.info(f"Validation {self.monitor} improved ({self.best_score:.4f} --> {val_metric:.4f}).")
                else: # mode == 'max'
                    logger.info(f"Validation {self.monitor} improved ({self.best_score:.4f} --> {val_metric:.4f}).")
            self.best_score = val_metric
            self.is_best_epoch = True
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} of {self.patience}. Current {self.monitor}: {val_metric:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True

