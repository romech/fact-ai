import torch
import pytorch_lightning as pl
import numpy as np
from collections import OrderedDict

from .networks.baseline import load_baseline_network
from .networks.complex import load_complex_network, RealToComplex
from .networks.complex.discriminator import Discriminator
from .networks.attack import UNet

class BaselineNetwork(pl.LightningModule):
    def __init__(self, checkpoint_path):
        super(BaselineNetwork, self).__init__()

        # Load pretrained checkpoint
        model_ckpt = torch.load(checkpoint_path)
        self.hparams = model_ckpt['hyper_parameters']

        # Initialize network with pretrained weights
        self.encoder, self.processor, self.decoder = load_baseline_network(
            self.hparams['arch'],
            self.hparams['num_classes'],
            self.hparams['resnet_variant'],
            self.hparams['additional_layers']
        )
        encoder_state_dict = {k[k.find('.')+1:]: v for k, v in model_ckpt['state_dict'].items() if 'encoder' in k} 
        self.encoder.load_state_dict(encoder_state_dict)
        processor_state_dict = {k[k.find('.')+1:]: v for k, v in model_ckpt['state_dict'].items() if 'processor' in k} 
        self.processor.load_state_dict(processor_state_dict)
        decoder_state_dict = {k[k.find('.')+1:]: v for k, v in model_ckpt['state_dict'].items() if 'decoder' in k} 
        self.decoder.load_state_dict(decoder_state_dict)


    def forward(self, x):
        x = self.encoder(x)
        x = self.processor(x)
        out = self.decoder(x)
        return out

    def setup(self, device: torch.device):
        self.freeze()

    def train(self, mode: bool):
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination


class BaselineEncoder(pl.LightningModule):
    def __init__(self, checkpoint_path):
        super(BaselineEncoder, self).__init__()

        # Load pretrained checkpoint
        model_ckpt = torch.load(checkpoint_path)
        state_dict = {k[k.find('.')+1:]: v for k, v in model_ckpt['state_dict'].items() if 'encoder' in k} # Only need encoder weights
        self.hparams = model_ckpt['hyper_parameters']

        # Initialize network with pretrained weights
        self.encoder, _, _ = load_baseline_network(
            self.hparams['arch'],
            self.hparams['num_classes'],
            self.hparams['resnet_variant'],
            self.hparams['additional_layers']
        )
        self.encoder.load_state_dict(state_dict)

    def add_noise(self, a, gamma):
        epsilon = torch.normal(a.mean(0), torch.ones(a.shape[1:], device=a.device)).unsqueeze(0)
        return a + epsilon*gamma

    def forward(self, x):
        x = self.encoder(x)
        if self.hparams.noisy:
            x = self.add_noise(x, self.hparams.gamma)
        return x

    def setup(self, device: torch.device):
        self.freeze()

    def train(self, mode: bool):
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

class ComplexEncoder(pl.LightningModule):
    def __init__(self, checkpoint_path, drop_imag=True):
        super(ComplexEncoder, self).__init__()
        self.drop_imag = drop_imag

        # Load pretrained checkpoint
        model_ckpt = torch.load(checkpoint_path)
        state_dict = {k[k.find('.')+1:]: v for k, v in model_ckpt['state_dict'].items() if 'encoder' in k} # Only need encoder weights

        self.hparams = model_ckpt['hyper_parameters']

        # Initialize network with pretrained weights
        self.encoder, _, _ = load_complex_network(
            self.hparams['arch'],
            self.hparams['num_classes'],
            self.hparams['resnet_variant'],
        )
        self.encoder.load_state_dict(state_dict)
        self.realtocomplex = RealToComplex()

    def forward(self, x):
        a = self.encoder(x)

        with torch.no_grad():
            indices = np.random.permutation(a.size(0))
            b = a[indices]

        x, theta = self.realtocomplex(a, b)

        return x, theta

    def setup(self, device: torch.device):
        self.freeze()

    def train(self, mode: bool):
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

class AngleDiscriminator(pl.LightningModule):
    def __init__(self, checkpoint_path, in_size):
        super(AngleDiscriminator, self).__init__()

        # Load pretrained checkpoint
        model_ckpt = torch.load(checkpoint_path)
        state_dict = {k[k.find('.')+1:]: v for k, v in model_ckpt['state_dict'].items()}

        # Initialize network with pretrained weights
        in_size[1] *= 2
        self.angle_discriminator = Discriminator(in_size[1:])
        self.angle_discriminator.load_state_dict(state_dict)

    def forward(self, x):
        pred_theta = self.angle_discriminator(
            torch.cat([x[:, 0], x[:, 1]], dim=1)).squeeze(1)
        return pred_theta

    def setup(self, device: torch.device):
        self.freeze()

    def train(self, mode: bool):
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

class InversionNetwork(pl.LightningModule):
    def __init__(self, checkpoint_path, channels):
        super(InversionNetwork, self).__init__()

        # Load pretrained checkpoint
        model_ckpt = torch.load(checkpoint_path)
        state_dict = {k[k.find('.')+1:]: v for k, v in model_ckpt['state_dict'].items()}

        self.hparams = model_ckpt['hyper_parameters']

        # Initialize network with pretrained weights
        self.inversion_network = UNet(
            in_channels=channels,
            out_size=self.hparams.dims[1:]
        )
        self.inversion_network.load_state_dict(state_dict)

    def forward(self, x):
        return self.inversion_network(x)

    def setup(self, device: torch.device):
        self.freeze()

    def train(self, mode: bool):
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination
