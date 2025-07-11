# src/models/base_model.py

import torch
import torch.nn as nn
from src.utils.config_parser import load_config

from src.models.feature_extractors.resnet18 import EEGResNet18 

from src.models.heads.classification_head import ClassificationHead


class BaseModel(nn.Module):
    """
    组合特征提取器和模型头的顶层模型。
    根据配置文件动态加载和初始化特征提取器和模型头。
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): 完整的实验配置字典，包含 model 相关的配置信息。
                           例如 config['model']['feature_extractor'] 和 config['model']['head']。
        """
        super(BaseModel, self).__init__()
        self.config = config

        # 1. 初始化特征提取器
        fe_cfg = config['model']['feature_extractor']
        if fe_cfg['name'] == 'ResNet18':
            self.feature_extractor = EEGResNet18(
                in_channels=fe_cfg['in_channels'],
                embedding_dim=fe_cfg['embedding_dim']
            )
        else:
            raise ValueError(f"Unknown feature extractor specified in config: {fe_cfg['name']}")

        # 2. 初始化模型头
        head_cfg = config['model']['head']
        head_input_dim = fe_cfg['embedding_dim'] 
        
        if head_cfg['type'] == 'classification':
            self.head = ClassificationHead(
                input_dim=head_input_dim,
                num_classes=head_cfg['num_classes']
            )
        else:
            raise ValueError(f"Unknown model head type specified in config: {head_cfg['type']}")

        print(f"Initialized BaseModel with Feature Extractor: {fe_cfg['name']} and Head: {head_cfg['type']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 原始 EEG 输入数据，形状为 (batch_size, in_channels, time_points)。
        Returns:
            torch.Tensor: 模型的最终输出 (例如分类 logits 或回归预测)。
        """
        features = self.feature_extractor(x)
        output = self.head(features)
        return output

