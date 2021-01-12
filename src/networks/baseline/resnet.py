import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, downsample):
        super(ResidualBlock, self).__init__()

        self.downsample = downsample
        self.channels = channels

        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2 if downsample else 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        if self.downsample:
            out = self.network + F.pad(x[..., ::2, ::2], (0, 0, 0, 0, self.channels // 2, self.channels // 2))
        else:
            out = self.network + x

        return F.relu(out)


class ResNetEncoder(nn.Module):
    def __init__(self, n, in_nc):
        super(ResNetEncoder, self).__init__()

        network = [
            nn.Conv2d(in_nc, 16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ]

        for i in range(n):
            network.append(ResidualBlock(16, False))

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
        return self.linear(out)
