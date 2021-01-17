import pytorch_lightning as pl
from resnet import *
from vgg import *
from alexnet import *
from lenet import *


class BaselineModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, resnet_variant="alpha", n=3):
        """
        :param model_name: String denoting which baseline network to use.
        :param num_classes: Number of classes in the dataset.
        :param resnet_variant: String defining the split between processor and decoder in the ResNet network.
            Either 'alpha' or 'beta'.
        :param n: Integer determining number of layers in the ResNet network. The number of layers will be 6n + 2.
        """
        super(BaselineModel, self).__init__()
        self.save_hyperparameters()

        self.rotation = RealToComplex()
        self.inverse_rotation = ComplexToReal()

        if model_name == "VGG":
            self.encoder = VGGEncoderComplex()
            self.processor = VGGProcessorComplex()
            self.decoder = VGGDecoderComplex(num_classes)
        elif model_name == "ResNet":
            self.encoder = ResNetEncoderComplex(n)
            self.processor = ResNetProcessorComplex(n, resnet_variant)
            self.decoder = ResNetDecoderComplex(n, num_classes, resnet_variant)
        elif model_name == "AlexNet":
            self.encoder = AlexNetEncoderComplex()
            self.processor = AlexNetProcessorComplex()
            self.decoder = AlexNetDecoderComplex(num_classes)
        else:
            self.encoder = LeNetEncoderComplex()
            self.processor = LeNetProcessorComplex(num_classes)
            self.decoder = LeNetDecoderComplex()

    def training_step(self, batch, batch_idx):
        input, target = batch

        batch_size = input.shape[0]

        a = self.encoder(input[:batch_size//2])
        b = self.encoder(input[batch_size//2:])
        complex_tensor, theta = self.rotation(a, b)
        complex_tensor = self.processor(complex_tensor)
        out = self.inverse_rotation(complex_tensor, theta)
        out = self.decoder(out)

        return F.cross_entropy(out, target)

