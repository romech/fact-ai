import torch.nn as nn
import torch.nn.functional as F
import torch


class ResidualBlock(nn.Module):
    def __init__(self, channels, downsample):
        super(ResidualBlock, self).__init__()

        self.downsample = downsample
        self.channels = channels

        self.network = nn.Sequential(
            nn.Conv2d(channels // 2 if downsample else channels, channels, kernel_size=3, padding=1,
                      stride=2 if downsample else 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        if self.downsample:
            out = self.network(x) + F.pad(x[..., ::2, ::2], (0, 0, 0, 0, self.channels // 4, self.channels // 4))
        else:
            out = self.network(x) + x

        return F.relu(out)


class ResNetEncoder(nn.Module):
    def __init__(self, n, additional_layers=False):
        super(ResNetEncoder, self).__init__()

        network = [
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ]

        for i in range(n):
            network.append(ResidualBlock(16, False))

        if additional_layers:
            network += [
                nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True)
            ]

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class ResNetProcessor(nn.Module):
    def __init__(self, n, variant="alpha"):
        super(ResNetProcessor, self).__init__()

        network = [ResidualBlock(32, True)]
        for i in range(n - 1):
            network.append(ResidualBlock(32, False))
        network.append(ResidualBlock(64, True))

        if variant == "beta":
            for i in range(n - 2):
                network.append(ResidualBlock(64, False))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class ResNetDecoder(nn.Module):
    def __init__(self, n, num_classes, variant="alpha"):
        super(ResNetDecoder, self).__init__()

        conv_layers = [ResidualBlock(64, False)]
        if variant == "alpha":
            for i in range(n - 2):
                conv_layers.append(ResidualBlock(64, False))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.mean([2, 3])          # global average pooling
        return self.linear(torch.flatten(out, 1))


def resnet20(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(3, additional_layers), \
        ResNetProcessor(3, variant),\
        ResNetDecoder(3, num_classes, variant)

def resnet32(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(5, additional_layers), \
        ResNetProcessor(5, variant), \
        ResNetDecoder(5, num_classes, variant)

def resnet44(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(7, additional_layers), \
        ResNetProcessor(7, variant), \
        ResNetDecoder(7, num_classes, variant)

def resnet56(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(9, additional_layers), \
        ResNetProcessor(9, variant), \
        ResNetDecoder(9, num_classes, variant)

def resnet110(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(18, additional_layers), \
        ResNetProcessor(18, variant), \
        ResNetDecoder(18, num_classes, variant)
