from ..baseline.resnet import *
from .complex_layers import *


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

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.mean([2, 3])          # global average pooling
        return self.linear(torch.flatten(out, 1))


def resnet20complex(num_classes, variant='alpha'):
    return \
        ResNetEncoderComplex(3), \
        ResNetProcessorComplex(3, variant),\
        ResNetDecoderComplex(3, num_classes, variant)

def resnet32complex(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoderComplex(5), \
        ResNetProcessorComplex(5, variant), \
        ResNetDecoderComplex(5, num_classes, variant)

def resnet44complex(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoderComplex(7), \
        ResNetProcessorComplex(7, variant), \
        ResNetDecoderComplex(7, num_classes, variant)

def resnet56complex(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoderComplex(9), \
        ResNetProcessorComplex(9, variant), \
        ResNetDecoderComplex(9, num_classes, variant)

def resnet110complex(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoderComplex(18), \
        ResNetProcessorComplex(18, variant), \
        ResNetDecoderComplex(18, num_classes, variant)
