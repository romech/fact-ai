import torch.nn as nn


class VGGEncoder(nn.Module):
    def __init__(self, additional_layers=False):
        super(VGGEncoder, self).__init__()

        network = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True)
        ]

        if additional_layers:
            network += [
                nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True)
            ]

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class VGGProcessor(nn.Module):
    def __init__(self):
        super(VGGProcessor, self).__init__()

        self.network = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.network(x)


class VGGDecoder(nn.Module):
    def __init__(self, num_classes):
        super(VGGDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def vgg(num_classes, additional_layers=False):
    return \
        VGGEncoder(additional_layers), \
        VGGProcessor(), \
        VGGDecoder(num_classes)