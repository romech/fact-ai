import pytorch_lightning as pl
from resnet import *
from vgg import *
from alexnet import *
from lenet import *
import torch


class BaselineModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, resnet_variant="alpha", n=3, noisy=False, additional_layers=False,
                 gamma=1.0):
        """
        :param model_name: String denoting which baseline network to use.
        :param num_classes: Number of classes in the dataset.
        :param resnet_variant: String defining the split between processor and decoder in the ResNet network.
            Either 'alpha' or 'beta'.
        :param n: Integer determining number of layers in the ResNet network. The number of layers will be 6n + 2.
        :param noisy: Boolean denoting whether noise should be added to the output of the encoder.
        :param additional_layers: Boolean determining whether an additional conv-layer should be attached to the encoder.
        :param gamma: Noise parameter for the noisy baseline.
        """
        super(BaselineModel, self).__init__()
        self.save_hyperparameters()

        if model_name == "VGG":
            self.encoder = VGGEncoder(additional_layers)
            self.processor = VGGProcessor()
            self.decoder = VGGDecoder(num_classes)
        elif model_name == "ResNet":
            self.encoder = ResNetEncoder(n, additional_layers)
            self.processor = ResNetProcessor(n, resnet_variant)
            self.decoder = ResNetDecoder(n, num_classes, resnet_variant)
        elif model_name == "AlexNet":
            self.encoder = AlexNetEncoder(additional_layers)
            self.processor = AlexNetProcessor()
            self.decoder = AlexNetDecoder(num_classes)
        else:
            self.encoder = LeNetEncoder(additional_layers)
            self.processor = LeNetProcessor(num_classes)
            self.decoder = LeNetDecoder()

    def add_noise(self, a, gamma):
        epsilon = torch.randn(a.shape)          # not sure about this one
        return a + epsilon*gamma

    def training_step(self, batch, batch_idx):
        input, target = batch

        out = self.encoder(input)
        if self.hparams.noisy:
            out = self.add_noise(out, self.hparams.gamma)
        out = self.processor(out)
        out = self.decoder(out)

        return F.cross_entropy(out, target)

