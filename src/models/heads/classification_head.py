# src/models/heads/classification_head.py

import torch
import torch.nn as nn

class ClassificationHead:
    def __init__(self, num_classes, embedding_dim=512):
        """
        Args:
            num_classes: 分类任务的类别数量。
            embedding_dim: 特征向量的维度，通常与特征提取器输出的维度一致。
        """
        super(ClassificationHead, self).__init__()        
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: 输入特征向量，形状为 (batch_size, embedding_dim)。
        
        Returns:
            logits: 分类结果，形状为 (batch_size, num_classes)。
        """
        # 假设 x 是经过特征提取器处理后的特征向量
        logits = self.fc(x)
        return logits
