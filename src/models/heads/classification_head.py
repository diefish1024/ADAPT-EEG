# src/models/heads/classification_head.py

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        """
        Args:
            input_dim: The dimension of the feature vector from the feature extractor.
            num_classes: The number of categories for the classification task.
        """
        super(ClassificationHead, self).__init__()        
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input feature vector, (batch_size, input_dim).
        
        Returns:
            logits: classification results, (batch_size, num_classes).
        """
        logits = self.fc(x)
        return logits

