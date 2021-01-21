import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError
import torch
from torchvision.utils import save_image, make_grid
from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict

from src.networks.baseline import load_baseline_network
from src.networks.complex import load_complex_network, RealToComplex
from src.networks.complex.discriminator import Discriminator
from src.networks.attack import UNet
from src.utils import get_encoder_output_size
import numpy as np
from pytorch_lightning.metrics.classification import Accuracy


class BaselineNetwork(pl.LightningModule):
    def __init__(self, checkpoint_path):
        super(BaselineNetwork, self).__init__()

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

    def forward(self, x):
        return self.encoder(x)

    def setup(self, device: torch.device):
        self.freeze()

    def train(self, mode: bool):
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

class ComplexNetwork(pl.LightningModule):
    def __init__(self, checkpoint_path):
        super(ComplexNetwork, self).__init__()

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
        x, _ = self.realtocomplex(a)
        out = x[:,0] # Drop imaginary part
        print(out.size())
        exit()
        return out

    def setup(self, device: torch.device):
        self.freeze()

    def train(self, mode: bool):
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination


# Number of feature channels ouputted from each network's encoder
encoder_features = {
    'resnet20': 16,
    'resnet32': 16,
    'resnet44': 16,
    'resnet56': 16,
    'resnet110': 16,
    'lenet': 6,
    'alexnet': 384,
    'vgg': 256
}

class AngleDiscriminator(pl.LightningModule):
    def __init__(
            self,
            weights,
            dims,
            lr=0.001,
            schedule='none',
            steps=[100,150],
            step_factor=0.1,
    ):
        super(AngleDiscriminator, self).__init__()
        self.save_hyperparameters()

        self.encoder = BaselineNetwork(weights)
        self.encoder.freeze()

        channels = get_encoder_output_size(self.encoder, dims)[0]
        dims[0] = 2*channels
        self.discriminator = Discriminator(dims)
        self.rotation = RealToComplex()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        print(self.hparams)

    def forward(self, x):
        out = self.discriminator(torch.cat([x[:, 0], x[:, 1]], dim=1))
        return out

    def training_step(self, batch, batch_idx):
        x, _ = batch

        a = self.encoder(x)

        with torch.no_grad():
            indices = np.random.permutation(a.size(0))
            b = a[indices]

        x, theta = self.rotation(a, b)
        pred = self(x)

        # Calculate loss and reconstruction error
        loss = F.mse_loss(pred, theta)
        mae = self.train_mae(pred, theta)
        self.log('train_loss', loss)
        self.log('train_mae', mae)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        a = self.encoder(x)

        with torch.no_grad():
            indices = np.random.permutation(a.size(0))
            b = a[indices]

        x, theta = self.rotation(a, b)
        pred = self(x)

        # Calculate loss and reconstruction error
        loss = F.mse_loss(pred, theta)
        mae = self.val_mae(pred, theta)
        self.log('val_loss', loss)
        self.log('val_mae', mae)

        return OrderedDict({
            'val_mae': mae,
            'val_loss': loss,
            'val_sample': torch.cat([x[:10], pred[:10]], dim=3) if batch_idx==0 else None
        })

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(
            self.inversion_network.parameters(),
            lr=self.hparams.lr,
        )

        # Define schedule
        if self.hparams.schedule == 'step':
            schedule = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.steps,
                gamma=self.hparams.step_factor,
            )
            return [optimizer], [schedule]
        else:
            return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument( '--experiment_name', type=str, help='Name of experiment.', default='default')
        parser.add_argument( '--data_path', type=Path, help='Path to download data.', default='data/')
        parser.add_argument( '--output_path', type=Path, help='Path to save output.', default='output/')
        parser.add_argument( '--weights', type=Path, help='Path to pretrained network checkpoint.', required=True)
        parser.add_argument('--seed', type=int, help='Seed to allow for reproducible results.', default=0)

        parser.add_argument('--dataset', type=str, help='cifar10 | cifar100 | celeba | cub200', default='cifar10')
        parser.add_argument('--workers', type=int, help='Number of dataloader workers.', default=6)
        parser.add_argument('--batch_size', type=int, help='Number of per batch samples.', default=64)
        parser.add_argument('--lr', type=float, help='Learning rate.', default=0.001)
        parser.add_argument('--schedule', type=str, help='Learning rate schedule (none | step)', default='none')
        parser.add_argument('--steps', nargs='+', type=int, help='Epochs where LR is reduced in step schedule.', default=[100,150] )
        parser.add_argument('--step_factor', type=float, help='Step reduction rate in step schedule .', default=0.1)

        return parser


class InferenceAttack1(pl.LightningModule):
    def __init__(self, attacker_ckpt, complex=False, prototype_ckpt=None):
        super(InferenceAttack1, self).__init__()

        self.prototype_net = None
        self.attacker = BaselineNetwork(attacker_ckpt)
        self.attacker.freeze()

        if prototype_ckpt and not complex:
            # for real-valued networks we can use inversion attack 2 (or 1 depending on how we define it)
            self.prototype_net = Inversion2Model.load_from_checkpoint(prototype_ckpt)
        elif prototype_ckpt and complex:
            # for complex networks should use inversion attack 1
            self.prototype_net = Inversion1Model.load_from_checkpoint(prototype_ckpt)

        self.acc = Accuracy()

    def forward(self, x):
        if self.prototype_net:
            x = self.prototype_net(x)

        return self.attacker(x)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        acc = self.acc(pred, y)

        self.log("test_acc", acc)


class Inversion1Model(pl.LightningModule):
    def __init__(self,
                 weights,
                 img_size,
                 lr=0.001,
                 schedule='none',
                 steps=[100, 150],
                 step_factor=0.1
                 ):
        super(Inversion1Model, self).__init__()

        self.encoder = BaselineNetwork(weights)
        self.encoder.freeze()

        if "complex" in self.encoder.hparams.arch:
            self.rotation = RealToComplex()
            self.discriminator = AngleDiscriminator.load_from_checkpoint(weights)
            self.discriminator.freeze()

        self.inversion_network = UNet(
            encoder_features[self.encoder.hparams.arch],
            img_size[1:]
        )

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        print(self.hparams)

    def forward(self, x):
        feats = self.encoder(x) # Encoder features

        if "complex" in self.encoder.hparams.arch:
            encoded_feats = self.rotation(x, torch.randn(feats.shape, device=feats.device))
            feats = self.discriminator(encoded_feats)

        out = self.inversion_network(feats) # Reconstruction of x
        return out

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred = self(x)

        # Calculate loss and reconstruction error
        loss = F.mse_loss(pred, x)
        mae = self.train_mae(pred, x)
        self.log('train_loss', loss)
        self.log('train_mae', mae)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and reconstruction error
        loss = F.mse_loss(pred, x)
        mae = self.val_mae(pred, x)
        self.log('val_loss', loss)
        self.log('val_mae', mae)

        return OrderedDict({
            'val_mae': mae,
            'val_loss': loss,
            'val_sample': torch.cat([x[:10], pred[:10]], dim=3) if batch_idx==0 else None
        })

    def validation_epoch_end(self, outputs):
        # Visualize reconstruction
        grid = make_grid(outputs[0]['val_sample'], nrow=2, normalize=True)
        tensorboard = self.logger.experiment
        tensorboard.add_image('val_samples', grid, self.current_epoch)

        # Print validation results
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['val_mae'] for x in outputs]).mean()
        print(f'Validation Loss: {avg_loss} Validation MAE: {avg_mae}')

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(
            self.inversion_network.parameters(),
            lr=self.hparams.lr,
        )

        # Define schedule
        if self.hparams.schedule == 'step':
            schedule = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.steps,
                gamma=self.hparams.step_factor,
            )
            return [optimizer], [schedule]
        else:
            return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--experiment_name', type=str, help='Name of experiment.', default='default')
        parser.add_argument('--data_path', type=Path, help='Path to download data.', default='data/')
        parser.add_argument('--output_path', type=Path, help='Path to save output.', default='output/')
        parser.add_argument('--weights', type=Path, help='Path to pretrained network checkpoint.', required=True)
        parser.add_argument('--complex', action='store_true', help='Loading a complex pretrained network?')

        parser.add_argument('--dataset', type=str, help='cifar10 | cifar100 | celeba | cub200', default='cifar10')
        parser.add_argument('--workers', type=int, help='Number of dataloader workers.', default=6)
        parser.add_argument('--batch_size', type=int, help='Number of per batch samples.', default=64)
        parser.add_argument('--lr', type=float, help='Learning rate.', default=0.001)
        parser.add_argument('--schedule', type=str, help='Learning rate schedule (none | step)', default='none')
        parser.add_argument('--steps', nargs='+', type=int, help='Epochs where LR is reduced in step schedule.', default=[100,150] )
        parser.add_argument('--step_factor', type=float, help='Step reduction rate in step schedule .', default=0.1)

        return parser


class Inversion2Model(pl.LightningModule):
    def __init__(
            self, 
            weights,
            dims,
            complex=False,
            lr=0.001,
            schedule='none',
            steps=[100,150],
            step_factor=0.1,
        ):
        super(Inversion2Model, self).__init__()
        self.save_hyperparameters()

        self.encoder = BaselineNetwork(weights)
        self.encoder.freeze()
        channels = get_encoder_output_size(self.encoder, dims)[0]
        self.inversion_network = UNet(
            encoder_features[self.encoder.hparams.arch], 
            dims[1:]
        )

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        print(self.hparams)

    def forward(self, x):
        feats = self.encoder(x) # Encoder features 
        out = self.inversion_network(feats) # Reconstruction of x
        return out

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred = self(x)

        # Calculate loss and reconstruction error
        loss = F.mse_loss(pred, x)
        mae = self.train_mae(pred, x)
        self.log('train_loss', loss)
        self.log('train_mae', mae)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and reconstruction error
        loss = F.mse_loss(pred, x)
        mae = self.val_mae(pred, x)
        self.log('val_loss', loss)
        self.log('val_mae', mae)

        return OrderedDict({
            'val_mae': mae,
            'val_loss': loss,
            'val_sample': torch.cat([x[:10], pred[:10]], dim=3) if batch_idx==0 else None
        })

    def validation_epoch_end(self, outputs):
        # Visualize reconstruction
        grid = make_grid(outputs[0]['val_sample'], nrow=2, normalize=True)
        tensorboard = self.logger.experiment
        tensorboard.add_image('val_samples', grid, self.current_epoch)
        
        # Print validation results
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['val_mae'] for x in outputs]).mean()
        print(f'Validation Loss: {avg_loss} Validation MAE: {avg_mae}')

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(
            self.inversion_network.parameters(),
            lr=self.hparams.lr,
        )

        # Define schedule
        if self.hparams.schedule == 'step':
            schedule = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.steps,
                gamma=self.hparams.step_factor,
            )
            return [optimizer], [schedule]
        else:
            return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument( '--experiment_name', type=str, help='Name of experiment.', default='default')
        parser.add_argument( '--data_path', type=Path, help='Path to download data.', default='data/')
        parser.add_argument( '--output_path', type=Path, help='Path to save output.', default='output/')
        parser.add_argument( '--weights', type=Path, help='Path to pretrained network checkpoint.', required=True)
        parser.add_argument('--complex', action='store_true', help='Loading a complex pretrained network?.')
        parser.add_argument('--seed', type=int, help='Seed to allow for reproducible results.', default=0)

        parser.add_argument('--dataset', type=str, help='cifar10 | cifar100 | celeba | cub200', default='cifar10')
        parser.add_argument('--workers', type=int, help='Number of dataloader workers.', default=6)
        parser.add_argument('--batch_size', type=int, help='Number of per batch samples.', default=64)
        parser.add_argument('--lr', type=float, help='Learning rate.', default=0.001)
        parser.add_argument('--schedule', type=str, help='Learning rate schedule (none | step)', default='none')
        parser.add_argument('--steps', nargs='+', type=int, help='Epochs where LR is reduced in step schedule.', default=[100,150] )
        parser.add_argument('--step_factor', type=float, help='Step reduction rate in step schedule .', default=0.1)
        
        return parser