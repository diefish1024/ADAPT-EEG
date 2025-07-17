# src/losses/classification_losses.py
import torch.nn as nn
 
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Standard Cross Entropy Loss for classification tasks."""
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)