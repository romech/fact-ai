import torch.nn as nn


class VGGEncoder(nn.Module):
    def __init__(self, in_nc):
        super(VGGEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_nc, 64, kernel_size=3, padding=1, stride=1),
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
        )

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
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
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
    def __init__(self, in_dim, num_classes):
        super(VGGDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(in_dim, num_classes)
       )

    def forward(self, x):
        return self.network(x)
