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
import numpy as np

from src.networks.baseline import load_baseline_network
from src.networks.complex import load_complex_network, RealToComplex, ComplexToReal
from src.networks.complex.discriminator import Discriminator
from src.networks.attack import UNet
from src.utils import get_encoder_output_size
import numpy as np


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
    def __init__(self, checkpoint_path, drop_imag=True):
        super(ComplexNetwork, self).__init__()
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

        x, theta = self.realtocomplex(a,b)

        if self.drop_imag:
            out = x[:,0] # Drop imaginary part
        else:
            out = x

        return out, theta

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

        self.hparams = model_ckpt['hyper_parameters']

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

class AngleInversionModel(pl.LightningModule):
    def __init__(self,
                 weights,
                 dims,
                 lr=0.001,
                 schedule='none',
                 steps=[10],
                 step_factor=0.1
        ):
        super(AngleInversionModel, self).__init__()
        self.save_hyperparameters()

        # Load pretrained encoder
        self.encoder = ComplexNetwork(weights, drop_imag=False)
        self.encoder.freeze()
        
        # Initialize attack models
        in_size = get_encoder_output_size(self.encoder, dims)
        in_size[1] *= 2
        self.angle_discriminator = Discriminator(in_size[1:])

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        print(self.hparams)

    def forward(self, x):
        feats, theta = self.encoder(x) # Encoder features
        pred_theta = self.angle_discriminator(
            torch.cat([feats[:, 0], feats[:, 1]], dim=1)).squeeze(1) # Predict angle
        return pred_theta, theta

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred_theta, theta = self(x)

        # Angle prediction loss and error
        loss = F.mse_loss(pred_theta, theta)
        mae = self.train_mae(pred_theta, theta)

        # Log
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae', mae)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred_theta, theta = self(x)

        # Angle prediction loss and error
        loss = F.mse_loss(pred_theta, theta)
        mae = self.val_mae(pred_theta, theta)

        # Log
        self.log('val_loss', loss)
        self.log('val_mae', mae)

        return OrderedDict({
            'val_mae': mae,
            'val_loss': loss,
        })

    def validation_epoch_end(self, outputs):
        # Print validation results
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['val_mae'] for x in outputs]).mean()
        print(f'Validation Loss: {avg_loss} Validation MAE: {avg_mae}')

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(
            #parameters,
            self.angle_discriminator.parameters(),
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
        parser.add_argument('--encoder_weights', type=Path, help='Path to pretrained network checkpoint.', required=True)
        parser.add_argument('--seed', type=int, help='Seed to allow for reproducible results.', default=0)

        parser.add_argument('--dataset', type=str, help='cifar10 | cifar100 | celeba | cub200', default='cifar10')
        parser.add_argument('--workers', type=int, help='Number of dataloader workers.', default=6)
        parser.add_argument('--batch_size', type=int, help='Number of per batch samples.', default=64)
        parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0001)
        parser.add_argument('--schedule', type=str, help='Learning rate schedule (none | step)', default='none')
        parser.add_argument('--steps', nargs='+', type=int, help='Epochs where LR is reduced in step schedule.', default=[15,25] )
        parser.add_argument('--step_factor', type=float, help='Step reduction rate in step schedule .', default=0.1)

        return parser

class FeatureInversionAngleModel(pl.LightningModule):
    def __init__(self,
                 encoder_weights,
                 angle_dis_weights,
                 dims,
                 lr=0.001,
                 schedule='none',
                 steps=[10],
                 step_factor=0.1
        ):
        super(FeatureInversionAngleModel, self).__init__()
        self.save_hyperparameters()

        # Load pretrained encoder
        self.encoder = ComplexNetwork(encoder_weights, drop_imag=False)
        self.encoder.freeze()
        self.complex_to_real = ComplexToReal()

        # Load pretrained angle discriminator
        in_size = get_encoder_output_size(self.encoder, dims)
        self.angle_discriminator = AngleDiscriminator(angle_dis_weights, in_size)
        self.angle_discriminator.freeze()
        
        # Initialize attack models
        self.inversion_network = UNet(
            in_channels=in_size[1]//2, 
            out_size=dims[1:]
        )

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        print(self.hparams)

    def forward(self, x):
        feats, _ = self.encoder(x) # Encoder features
        pred_theta = self.angle_discriminator(feats) # Predict angle
        a_pred = self.complex_to_real(feats, pred_theta) # Revert to real using predicted angle        
        out = self.inversion_network(a_pred) # Reconstruct x
        return out

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred = self(x)

        # Reconstruction loss and error
        loss = F.mse_loss(pred, x)
        mae = self.train_mae(pred, x)

        # Log
        self.log('train_loss', loss)
        self.log('train_mae', mae)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred = self(x)

        # Reconstruction loss and error
        loss = F.mse_loss(pred, x)
        mae = self.val_mae(pred, x)

        # Log
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
            #parameters,
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
        parser.add_argument('--encoder_weights', type=Path, help='Path to pretrained network checkpoint.', required=True)
        parser.add_argument('--angle_dis_weights', type=Path, help='Path to pretrained angle discriminator checkpoint.', default='.')
        parser.add_argument('--seed', type=int, help='Seed to allow for reproducible results.', default=0)
        
        parser.add_argument('--dataset', type=str, help='cifar10 | cifar100 | celeba | cub200', default='cifar10')
        parser.add_argument('--workers', type=int, help='Number of dataloader workers.', default=6)
        parser.add_argument('--batch_size', type=int, help='Number of per batch samples.', default=64)
        parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0001)
        parser.add_argument('--schedule', type=str, help='Learning rate schedule (none | step)', default='none')
        parser.add_argument('--steps', nargs='+', type=int, help='Epochs where LR is reduced in step schedule.', default=[15,25] )
        parser.add_argument('--step_factor', type=float, help='Step reduction rate in step schedule .', default=0.1)

        return parser

class FeatureInversionModel(pl.LightningModule):
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
        super(FeatureInversionModel, self).__init__()
        self.save_hyperparameters()

        if complex:
            self.encoder = ComplexNetwork(weights)
        else:
            self.encoder = BaselineNetwork(weights)
        self.encoder.freeze()
        channels = get_encoder_output_size(self.encoder, dims)[0]
        self.inversion_network = UNet(
            in_channels=channels, 
            out_size=dims[1:]
        )

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        print(self.hparams)

    def forward(self, x):
        feats, _ = self.encoder(x) # Encoder features
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
        parser.add_argument( '--encoder_weights', type=Path, help='Path to pretrained network checkpoint.', required=True)
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