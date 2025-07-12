# src/losses/classification_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for multi-class classification.
    Standard loss for source domain pre-training tasks.
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0):
        """
        Initializes the CrossEntropyLoss.
 
        Args:
            weight (torch.Tensor, optional): Weights for each class. Defaults to None.
            ignore_index (int): Specifies a target value that is ignored. Defaults to -100.
            reduction (str): Reduction to apply to the output: 'none' | 'mean' | 'sum'. Defaults to 'mean'.
            label_smoothing (float): Label smoothing value to mitigate overfitting. Defaults to 0.0.
        """
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
 
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss.
 
        Args:
            inputs (torch.Tensor): Raw model outputs (logits) of shape (N, C), where C is the number of classes.
                                   E.g., (batch_size, num_classes).
            targets (torch.Tensor): True class labels (integer indices) of shape (N,).
                                    E.g., (batch_size).
 
        Returns:
            torch.Tensor: Computed cross-entropy loss value.
        """
        if inputs.dim() == 1 or inputs.dim() == 0:
            raise ValueError("Inputs dimension must be at least 2 for CrossEntropyLoss (batch_size, num_classes).")
        if targets.dim() != 1:
            raise ValueError("Targets dimension must be 1 (batch_size) for CrossEntropyLoss.")
 
        return self.cross_entropy(inputs, targets)

