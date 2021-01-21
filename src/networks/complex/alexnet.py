from ..baseline.alexnet import *
from .complex_layers import *


class AlexNetEncoderComplex(AlexNetEncoder):
    def __init__(self):
        super(AlexNetEncoderComplex, self).__init__(additional_layers=True)

    def forward(self, x):
        return self.network(x)


class AlexNetProcessorComplex(nn.Module):
    def __init__(self):
        super(AlexNetProcessorComplex, self).__init__()

        self.conv1 = Conv2dComplex(384, 384, kernel_size=3, padding=1, stride=1)
        self.conv2 = Conv2dComplex(384, 256, kernel_size=3, padding=1, stride=1)
        self.maxpool = MaxPool2dComplex(3, 2)

    def forward(self, x):
        out = activation_complex_dynamic(self.conv1(x))
        out = activation_complex_dynamic(self.conv2(out))
        return self.maxpool(out)


class AlexNetDecoderComplex(AlexNetDecoder):
    def __init__(self, num_classes):
        super(AlexNetDecoderComplex, self).__init__(num_classes)

    def forward(self, x):
        return self.network(x)

def alexnetcomplex(num_classes):
    return \
        AlexNetEncoderComplex(), \
        AlexNetProcessorComplex(), \
        AlexNetDecoderComplex(num_classes) 

