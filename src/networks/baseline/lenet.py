import torch.nn as nn


class LeNetEncoder(nn.Module):
    def __init__(self, in_nc):
        super(LeNetEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_nc, 6, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.network(x)


class LeNetProcessor(nn.Module):
    def __init__(self, lin_in_dim, num_classes):
        super(LeNetProcessor, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(lin_in_dim, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class LeNetDecoder(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(LeNetDecoder, self).__init__()

    def forward(self, x):
        return x
