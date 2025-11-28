'''
Orientierung an 
https://medium.com/@sainihith0130/fashion-mnist-classification-enhancing-fashion-mnist-classification-through-optimized-cnn-and-84a74eb2d3a1
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Block:
        x -> Conv2d(64,64) -> Conv2d(64,64) -> +x -> ReLU
    Alle Convs: kernel_size=3, padding=1, stride=1
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)
        return out


class SmallResNet(nn.Module):
    """
    Architektur:

    Input (N,1,28,28)
      -> Conv2d(1,64,3x3, padding=1)
      -> ResidualBlock
      -> ResidualBlock
      -> ResidualBlock
      -> GlobalAveragePooling
      -> Linear(64,64) + ReLU
      -> Linear(64,10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 28x28 -> 14x14
        )

        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # GlobalAveragePooling2D -> (N, 64)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



