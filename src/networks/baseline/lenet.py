import torch.nn as nn


class LeNetEncoder(nn.Module):
    def __init__(self, additional_layers=False):
        super(LeNetEncoder, self).__init__()

        network = [
            nn.Conv2d(3, 6, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        if additional_layers:
            network += [
                nn.Conv2d(6, 6, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True)
            ]

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class LeNetProcessor(nn.Module):
    def __init__(self, num_classes):
        super(LeNetProcessor, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class LeNetDecoder(nn.Module):
    def __init__(self):
        super(LeNetDecoder, self).__init__()

    def forward(self, x):
        return x


def lenet(num_classes, additional_layers=False):
    return \
        LeNetEncoder(additional_layers), \
        LeNetProcessor(num_classes), \
        LeNetDecoder()