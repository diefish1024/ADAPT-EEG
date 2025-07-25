# src/models/feature_extractors/eeg_resnet18.py

import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    expansion = 1 # for ResNet-18/34, output channels = input channels
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
class EEGResNet18(nn.Module): # Class name changed to EEGResNet18
    def __init__(self, in_channels: int = 62, embedding_dim: int = 256):
        """
        Args:
            in_channels: The number of EEG input channels (62 for SEED dataset raw EEG)
            embedding_dim: Dimensions of feature vector output
        """
        super(EEGResNet18, self).__init__()
        # Ensure 'block' and 'layers' are consistent with ResNet18 structure
        block = BasicBlock1D
        layers = [2, 2, 2, 2] # ResNet-18 specific layer configuration

        self.in_channels_current = 64 # out_channels of first conv, this will be updated in _make_layer
        self.output_dim = embedding_dim # This will be the final output dimension of the extractor
        
        # Initial convolutional layer
        self.conv1 = nn.Conv1d(in_channels, self.in_channels_current, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels_current)
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
        
        # Final fully connected layer to map to embedding_dim
        self.fc = nn.Linear(512 * block.expansion, self.output_dim)

        # initialize weights 
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: nn.Module, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Create a phase of ResNet
        """
        downsample = None
        # downsample the identity or adjust the number of channels
        if stride != 1 or self.in_channels_current != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels_current, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels_current, out_channels, stride, downsample))
        self.in_channels_current = out_channels * block.expansion # update input channels for next block

        # append remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels_current, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected input: (batch_size, in_channels, time_points)
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

# Factory function to create an instance with default ResNet18 configuration
def build_eeg_resnet18(in_channels: int = 62, embedding_dim: int = 256) -> EEGResNet18:
    return EEGResNet18(in_channels=in_channels, embedding_dim=embedding_dim)

