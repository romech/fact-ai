import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import Accuracy
import torch
from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
import numpy as np

from .networks.baseline import load_baseline_network
from .networks.complex import load_complex_network, RealToComplex, ComplexToReal, Discriminator
from .utils import get_encoder_output_size


class BaselineModel(pl.LightningModule):
    def __init__(
            self, 
            arch,
            num_classes,
            additional_layers=False,
            resnet_variant='alpha',
            noisy=False,
            gamma=1.0,
            optimizer='sgd',
            lr=0.1,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0001,
            momentum=0.9,
            schedule='none',
            steps=[100,150],
            step_factor=0.1,
        ):
        super(BaselineModel, self).__init__()
        self.save_hyperparameters()

        self.encoder, self.processor, self.decoder = load_baseline_network(
            arch,
            num_classes,
            resnet_variant,
            additional_layers
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        print(self.hparams)

    def forward(self, x):
        x = self.encoder(x)
        if self.hparams.noisy:
            x = self.add_noise(x, self.hparams.gamma)
        x = self.processor(x)
        out = self.decoder(x)
        return out

    def add_noise(self, a, gamma):
        epsilon = torch.normal(a, torch.ones(a.shape, device=a.device))
        return a + epsilon*gamma

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Pass through network
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.train_acc(pred, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.val_acc(pred, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return OrderedDict({
            'val_acc': acc,
            'val_loss': loss,
        })

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print(f'Validation Loss: {avg_loss} Validation Accuracy: {avg_acc}')

    def configure_optimizers(self):
        # Combine params of network parts together
        parameters = set()
        for net in [self.encoder, self.processor, self.decoder]:
            parameters |= set(net.parameters())

        # Define optimizer
        if self.hparams.optimizer == 'sgd':
            print('Using SGD optimizer...')
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay, 
            )
        elif self.hparams.optimizer == 'adam':
            print('Using Adam optimizer...')
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )
        else:
            raise NotImplementedError('{} is not an available optimzer'.format(self.hparams.optimizer))

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
        parser.add_argument('--seed', type=int, help='Seed to allow for reproducible results.', default=0)
        
        parser.add_argument('--dataset', type=str, help='cifar10 | cifar100 | celeba | cub200', default='cifar10')
        parser.add_argument('--workers', type=int, help='Number of dataloader workers.', default=6)
        parser.add_argument('--batch_size', type=int, help='Number of per batch samples.', default=128)
        parser.add_argument('--optimizer', type=str, help='sgd | adam', default='sgd')
        parser.add_argument('--lr', type=float, help='Learning rate.', default=0.1)
        parser.add_argument('--beta1', type=float, help='Adam beta 1 parameter.', default=0.9)
        parser.add_argument('--beta2', type=float, help='Adam beta 2 parameter.', default=0.999)
        parser.add_argument('--weight_decay', type=float, help='Weight decay.', default=0.0001)
        parser.add_argument('--momentum', type=float, help='SGD momentum.', default=0.9)
        parser.add_argument('--schedule', type=str, help='Learning rate schedule (none | step)', default='none')
        parser.add_argument('--steps', nargs='+', type=int, help='Epochs where LR is reduced in step schedule.', default=[100,150] )
        parser.add_argument('--step_factor', type=float, help='Step reduction rate in step schedule .', default=0.1)
        
        parser.add_argument('--arch', type=str, default='resnet20',
            help='Network architecture (resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | lenet | alexnet | vgg)', )
        parser.add_argument('--additional_layers', action='store_true', help='Add additional layers to network.')
        parser.add_argument('--resnet_variant', type=str, help='alpha | beta', default='alpha')
        parser.add_argument('--noisy', action='store_true', help='Add noise to encoder output.')
        parser.add_argument('--gamma', type=float, help='Noise scaling factor.', default=1.0)

        return parser

class ComplexModel(pl.LightningModule):
    def __init__(
            self, 
            arch,
            num_classes,
            dims,
            resnet_variant='alpha',
            optimizer='adam',
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0001,
            momentum=0.9,
            schedule='none',
            steps=[100,150],
            step_factor=0.1,
            k=5
        ):
        super(ComplexModel, self).__init__()
        self.save_hyperparameters()

        self.encoder, self.processor, self.decoder = load_complex_network(
            arch,
            num_classes,
            resnet_variant,
        )
        self.real_to_complex = RealToComplex()
        self.complex_to_real = ComplexToReal()
        size = get_encoder_output_size(self.encoder, dims)
        self.discriminator = Discriminator(size=size)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        print(self.hparams)

    def forward(self, inp):
        # Pass through encoder
        a = self.encoder(inp)

        # Shuffle batch elements of a to create b
        with torch.no_grad():
            indices = np.random.permutation(a.size(0))
            b = a[indices]

        # Convert to complex and pass through processor
        x, theta = self.real_to_complex(a, b)
        h = self.processor(x)

        # Convert back to real and get output
        y = self.complex_to_real(h, theta)
        out = self.decoder(y)

        return out, a

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        # Pass through classifier
        pred, a = self(x)

        # Get discriminator score expectation over k rotations
        score_fake = 0
        for k in range(self.hparams.k):
            # Shuffle batch to get b
            indices = np.random.permutation(a.size(0))
            b = a[indices]

            # Rotate a
            x, _ = self.real_to_complex(a, b) # Random rotation
            a_rotated = x[:,0] # Drop complex component

            # Get discriminator score  
            score_fake += self.discriminator(a_rotated)

        score_fake /= self.hparams.k # Average score

        # Generator step
        if optimizer_idx == 0:
            # Get adversarial loss on rotated features
            g_loss_adv = -torch.mean(score_fake)

            # Calculate classification cross entropy loss and accuracy
            g_loss_ce = F.cross_entropy(pred, y)
            acc = self.train_acc(pred, y)

            # Log
            self.log('train_loss_adv_g', g_loss_adv, prog_bar=True)
            self.log('train_loss_ce', g_loss_ce, prog_bar=True)
            self.log('train_acc', acc)

            return g_loss_ce+g_loss_adv

        # Discriminator step
        if optimizer_idx == 1:
            # Clip discriminator weights
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            
            # Get adversarial loss on rotated and unrotated features
            d_loss_adv = -torch.mean(self.discriminator(a)) + torch.mean(score_fake)
            self.log('train_loss_adv_d', d_loss_adv, prog_bar=True)

            return d_loss_adv

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Pass through model
        pred, _ = self(x)

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.val_acc(pred, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return OrderedDict({
            'val_acc': acc,
            'val_loss': loss,
        })

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print(f'Validation Loss: {avg_loss} Validation Accuracy: {avg_acc}')

    def configure_optimizers(self):
        # Combine params of network parts together
        parameters = set()
        for net in [self.encoder, self.processor, self.decoder]:
            parameters |= set(net.parameters())

        # Define optimizers
        if self.hparams.optimizer == 'sgd':
            print('Using SGD optimizer...')
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay, 
            )
            optimizer_d = torch.optim.SGD(
                self.discriminator.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay, 
            )

        elif self.hparams.optimizer == 'adam':
            print('Using Adam optimizer...')
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )
            optimizer_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )
        else:
            raise NotImplementedError('{} is not an available optimzer'.format(self.hparams.optimizer))

        # Define schedules
        if self.hparams.schedule == 'step':
            schedule = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.steps,
                gamma=self.hparams.step_factor,
            )
            schedule_d = lr_scheduler.MultiStepLR(
                optimizer_d,
                milestones=self.hparams.steps,
                gamma=self.hparams.step_factor,
            )
            return [optimizer, optimizer_d], [schedule, schedule_d]
        else:
            return [optimizer, optimizer_d]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument( '--experiment_name', type=str, help='Name of experiment.', default='default')
        parser.add_argument( '--data_path', type=Path, help='Path to download data.', default='data/')
        parser.add_argument( '--output_path', type=Path, help='Path to save output.', default='output/')
        parser.add_argument('--seed', type=int, help='Seed to allow for reproducible results.', default=0)
        
        parser.add_argument('--dataset', type=str, help='cifar10 | cifar100 | celeba | cub200', default='cifar10')
        parser.add_argument('--workers', type=int, help='Number of dataloader workers.', default=6)
        parser.add_argument('--batch_size', type=int, help='Number of per batch samples.', default=128)
        parser.add_argument('--optimizer', type=str, help='sgd | adam', default='adam')
        parser.add_argument('--lr', type=float, help='Learning rate.', default=0.001)
        parser.add_argument('--beta1', type=float, help='Adam beta 1 parameter.', default=0.9)
        parser.add_argument('--beta2', type=float, help='Adam beta 2 parameter.', default=0.999)
        parser.add_argument('--weight_decay', type=float, help='Weight decay.', default=0.0001)
        parser.add_argument('--momentum', type=float, help='SGD momentum.', default=0.9)
        parser.add_argument('--schedule', type=str, help='Learning rate schedule (none | step)', default='none')
        parser.add_argument('--steps', nargs='+', type=int, default=[100,150],
            help='Epochs where LR is reduced during step schedule. (Space separated list of integers)')
        parser.add_argument('--step_factor', type=float, help='Step reduction rate during step schedule.', default=0.1)
        
        parser.add_argument('--arch', type=str, default='resnet20',
            help='Network architecture (resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | lenet | alexnet | vgg)', )
        parser.add_argument('--resnet_variant', type=str, help='alpha | beta', default='alpha')
        parser.add_argument('--k', type=int, help='Number of rotations that are averaged over for GAN training.', default=5)

        return parser
