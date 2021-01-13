import torch.nn as nn


class AlexNetEncoder(nn.Module):
    def __init__(self, additional_layers=False):
        super(AlexNetEncoder, self).__init__()

        network = [
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, stride=1),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True)
        ]

        if additional_layers:
            network += [
                nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True)
            ]

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class AlexNetProcessor(nn.Module):
    def __init__(self):
        super(AlexNetProcessor, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        return self.network(x)


class AlexNetDecoder(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.network(x)
