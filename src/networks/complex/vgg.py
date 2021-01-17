from ..baseline.vgg import *
from ..complex_layers import *


class VGGEncoderComplex(VGGEncoder):
    def __init__(self):
        super(VGGEncoderComplex, self).__init__(additional_layers=True)

    def forward(self, x):
        return self.network(x)


class VGGProcessorComplex(nn.Module):
    def __init__(self):
        super(VGGProcessorComplex, self).__init__()
        self.network = nn.ModuleList([
            Conv2dComplex(256, 512, kernel_size=3, padding=1, stride=1),
            Conv2dComplex(512, 512, kernel_size=3, padding=1, stride=1),
            Conv2dComplex(512, 512, kernel_size=3, padding=1, stride=1),
            Conv2dComplex(512, 512, kernel_size=3, padding=1, stride=1),
            Conv2dComplex(512, 512, kernel_size=3, padding=1, stride=1),
            Conv2dComplex(512, 512, kernel_size=3, padding=1, stride=1)
        ])

        self.pool = MaxPool2dComplex(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.pool(x)
        for i, module in enumerate(self.network):
            if i == 3:
                out = self.pool(out)
            out = activation_complex_dynamic(module(out))

        return self.pool(out)


class VGGDecoderComplex(VGGDecoder):
    def __init__(self, num_classes):
        super(VGGDecoderComplex, self).__init__(num_classes)
        self.rotation = ComplexToReal()

    def forward(self, x):
        return self.network(x)
