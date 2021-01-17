from ..baseline.lenet import *
from ..complex_layers import *


class LeNetEncoderComplex(LeNetEncoder):
    def __init__(self):
        super(LeNetEncoderComplex, self).__init__(additional_layers=True)
        self.rotation = RealToComplex()

    def forward(self, x):
        return self.network(x)


class LeNetProcessorComplex(nn.Module):
    def __init__(self, num_classes):
        super(LeNetProcessorComplex, self).__init__()

        self.conv1 = Conv2dComplex(6, 16, kernel_size=5, stride=1)
        self.max_pool = MaxPool2dComplex(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(2*16*5*5, 120)     # due to complex/real dimension multiply no. of input features by 2
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x):
        pass


class LeNetDecoderComplex(LeNetDecoder):
    def __init__(self):
        super(LeNetDecoderComplex, self).__init__()
