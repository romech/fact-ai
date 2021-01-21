from ..baseline.lenet import *
from .complex_layers import *


class LeNetEncoderComplex(LeNetEncoder):
    def __init__(self):
        super(LeNetEncoderComplex, self).__init__(additional_layers=True)

    def forward(self, x):
        return self.network(x)


class LeNetProcessorComplex(nn.Module):
    def __init__(self, num_classes):
        super(LeNetProcessorComplex, self).__init__()

        self.conv1 = Conv2dComplex(6, 16, kernel_size=5, stride=1)
        self.max_pool = MaxPool2dComplex(kernel_size=2, stride=2)
        self.linear1 = LinearComplex(16*5*5, 120)
        self.linear2 = LinearComplex(120, 84)
        self.linear3 = LinearComplex(84, num_classes)

    def forward(self, x):
        out = self.max_pool(activation_complex_dynamic(self.conv1(x)))
        out = torch.flatten(out, 2)
        out = activation_complex_dynamic(self.linear1(out))
        out = activation_complex_dynamic(self.linear2(out))
        out = self.linear3(out)

        return out


class LeNetDecoderComplex(LeNetDecoder):
    def __init__(self):
        super(LeNetDecoderComplex, self).__init__()


def lenetcomplex(num_classes):
    return \
        LeNetEncoderComplex(), \
        LeNetProcessorComplex(num_classes), \
        LeNetDecoderComplex()