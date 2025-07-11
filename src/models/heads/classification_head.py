# src/models/heads/classification_head.py

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        """
        Args:
            num_classes: The number of categories for the classification task.
            embedding_dim: The dimension of the feature vector, the same as the dimension of the feature extractor output.
        """
        super(ClassificationHead, self).__init__()        
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: imput feature vector, (batch_size, embedding_dim)。
        
        Returns:
            logits: classfication results, (batch_size, num_classes)。
        """
        logits = self.fc(x)
        return logits
