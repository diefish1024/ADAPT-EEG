# src/models/feature_extractors/resnet18.py

import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    expansion = 1 # for ResNet-18/34, output channels = input channels
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample # 用于匹配维度，当 stride != 1 或 in_channels != out_channels 时
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# --- 定义 1D ResNet 类 ---
class ResNet18(nn.Module):
    def __init__(self, block, layers, in_channels=62, embedding_dim=512):
        """
        Args:
            block: 残差块类型 (BasicBlock1D)
            layers: 一个列表，指定每个阶段的残差块数量 (e.g., [2, 2, 2, 2] for ResNet-18)
            in_channels: 输入 EEG 信号的通道数，例如 62 用于 SEED 数据集原始 EEG。
                         如果输入是经过预处理的特征图 (例如 (Frequencies, Timepoints)), 
                         则 in_channels 对应 Frequencies 数量。
            embedding_dim: ResNet 特征提取器输出的特征向量维度。
        """
        super(ResNet18, self).__init__()
        self.in_channels = 64 # ResNet 的初始 out_channels
        self.embedding_dim = embedding_dim
        
        # 初始卷积层：处理输入 EEG 数据
        # 这里的 kernel_size 和 stride 需要根据 EEG 数据的实际时间点数和期望的下采样率进行调整
        # 例如，为了捕捉更长的时间依赖，可以使用更大的 kernel_size
        self.conv1 = nn.Conv1d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始池化层：进一步降低时间维度
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 构建 ResNet 的四个主要阶段
        # 每个 make_layer 会创建多个 BasicBlock1D
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化 (将 (batch_size, channels, timepoints) 变为 (batch_size, channels))
        self.avgpool = nn.AdaptiveAvgPool1d(1) 
        
        self.fc = nn.Linear(512 * block.expansion, embedding_dim)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        构建 ResNet 的一个阶段
        """
        downsample = None
        # 如果 stride 不为 1 或通道数不匹配，需要对 identity 进行下采样或调整通道数
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        # 第一个块可能需要进行下采样或通道调整
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion # 更新当前输入通道数

        # 添加剩余的块 (通常 stride=1)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns:
            x: (batch_size, embedding_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # (batch_size, 512, 1) -> (batch_size, 512)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平操作，将 (batch, channels, 1) 变为 (batch, channels)

        x = self.fc(x) # 映射到 embedding_dim

        return x

def EEGResNet18(in_channels=62, embedding_dim=256):
    """
    创建一个用于 EEG 特征提取的 ResNet-18 模型实例。
    """
    return ResNet18(BasicBlock1D, [2, 2, 2, 2], in_channels=in_channels, embedding_dim=embedding_dim)