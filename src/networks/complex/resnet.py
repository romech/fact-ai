from ..baseline.resnet import *
from complex_layers import *


class ResidualBlockComplex(nn.Module):
    def __init__(self, channels, downsample):
        super(ResidualBlockComplex, self).__init__()

        self.downsample = downsample
        self.channels = channels

        self.network = nn.Sequential(
            Conv2dComplex(channels // 2 if downsample else channels, channels, kernel_size=3, padding=1,
                          stride=2 if downsample else 1),
            BatchNormComplex(),
            ActivationComplex(),
            Conv2dComplex(channels, channels, kernel_size=3, padding=1, stride=1),
            BatchNormComplex()
        )

    def forward(self, x):
        if self.downsample:
            out = self.network(x) + F.pad(x[..., ::2, ::2], (0, 0, 0, 0, self.channels // 4, self.channels // 4))
        else:
            out = self.network(x) + x

        return activation_complex(out, 1)


class ResNetEncoderComplex(ResNetEncoder):
    def __init__(self, n):
        super(ResNetEncoderComplex, self).__init__(n, additional_layers=True)

    def forward(self, x):
        return self.network(x)


class ResNetProcessorComplex(nn.Module):
    def __init__(self, n, variant="alpha"):
        super(ResNetProcessorComplex, self).__init__()

        network = [ResidualBlockComplex(32, True)]
        for i in range(n - 1):
            network.append(ResidualBlockComplex(32, False))
        network.append(ResidualBlockComplex(64, True))

        if variant == "beta":
            for i in range(n - 2):
                network.append(ResidualBlockComplex(64, False))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class ResNetDecoderComplex(ResNetDecoder):
    def __init__(self, n, num_classes, variant="alpha"):
        super(ResNetDecoderComplex, self).__init__(n, num_classes, variant)
        self.rotation = ComplexToReal()

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.mean([2, 3])          # global average pooling
        return self.linear(torch.flatten(out, 1))
