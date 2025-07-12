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
        self.downsample = downsample # match dimensions when stride != 1 or in_channels != out_channels
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

# --- 1D ResNet Class ---
class ResNet18(nn.Module):
    def __init__(self, block, layers, in_channels=62, embedding_dim=512):
        """
        Args:
            block: BasicBlock1D
            layers: A list specifying the number of residual blocks at each stage (e.g., [2, 2, 2, 2] for ResNet-18)
            in_channels: The number of EEG input channels (62 for SEED dataset raw EEG)
            embedding_dim: Dimensions of feature vector output
        """
        super(ResNet18, self).__init__()
        self.in_channels = 64 # out_channels of ResNet
        self.embedding_dim = embedding_dim
        
        # Initial convolutional layer
        # The kernel_size and stride here need to be adjusted according to
        # the actual number of time points in the EEG data and the desired downsampling rate
        self.conv1 = nn.Conv1d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Initial pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 4 main phases
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling ((batch_size, channels, timepoints) -> (batch_size, channels))
        self.avgpool = nn.AdaptiveAvgPool1d(1) 
        
        self.fc = nn.Linear(512 * block.expansion, embedding_dim)

        # initialize weights 
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Create a phase of ResNet
        """
        downsample = None
        # downsample the identity or adjust the number of channels
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion # update input channels

        # append remaining blocks
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
        x = torch.flatten(x, 1) # (batch, channels, 1) -> (batch, channels)

        x = self.fc(x) # mapping to embedding_dim

        return x

def EEGResNet18(in_channels=62, embedding_dim=256):
    """
    Create a ResNet-18 model instance for EEG
    """
    return ResNet18(BasicBlock1D, [2, 2, 2, 2], in_channels=in_channels, embedding_dim=embedding_dim)