import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError
from pytorch_lightning.metrics.classification import Accuracy
import torch
from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
import numpy as np

from .networks.complex import ComplexToReal
from .utils import get_encoder_output_size
from .frozen_models import *


class InferenceAttack1Model(pl.LightningModule):
    def __init__(
        self, 
        classifier_weights, 
        encoder_weights,
        inversion_net_weights, 
        angle_dis_weights,
        dims
    ):
        super(InferenceAttack1Model, self).__init__()

        # Load classification network
        self.classifier = BaselineNetwork(classifier_weights)
        self.classifier.freeze()

        # Load complex encoder
        self.encoder = ComplexEncoder(encoder_weights, drop_imag=False)
        self.encoder.freeze()

        # Load feature inversion network
        in_size = get_encoder_output_size(self.encoder, dims)
        self.inversion_network = InversionNetwork(inversion_net_weights, in_size[1])
        self.inversion_network.freeze()

        # Load angle discriminator
        self.angle_discriminator = AngleDiscriminator(angle_dis_weights, in_size)
        self.angle_discriminator.freeze()

        self.complex_to_real = ComplexToReal()

        self.acc = Accuracy()

        print(self.hparam)

    def forward(self, x):
        # Reconstruct input
        feats, _ = self.encoder(x) # Encoder features
        pred_theta = self.angle_discriminator(feats) # Predict angle
        a_pred = self.complex_to_real(feats, pred_theta) # Revert to real using predicted angle        
        x_pred = self.inversion_network(a_pred) # Reconstruct x

        # Classify on reconstruction
        out = self.classifier(x_pred)

        return out

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.val_acc(pred, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument( '--experiment_name', type=str, help='Name of experiment.', default='default')
        parser.add_argument( '--data_path', type=Path, help='Path to download data.', default='data/')
        parser.add_argument( '--output_path', type=Path, help='Path to save output.', default='output/')
        parser.add_argument('--seed', type=int, help='Seed to allow for reproducible results.', default=0)
        parser.add_argument( '--classifier_weights', type=Path, help='Path to classification network checkpoint.', required=True)
        parser.add_argument( '--inversion_net_weights', type=Path, help='Path to feature inversion network checkpoint.', required=True)
        parser.add_argument( '--angle_dis_weights', type=Path, help='Path to angle discriminator checkpoint.', required=True)
        parser.add_argument( '--encoder_weights', type=Path, help='Path to complex encoder checkpoint.', required=True)
        
        parser.add_argument('--dataset', type=str, help='cifar10 | cifar100 | celeba | cub200', default='cifar10')
        parser.add_argument('--workers', type=int, help='Number of dataloader workers.', default=6)
        parser.add_argument('--batch_size', type=int, help='Number of per batch samples.', default=64)
        
        return parser

class InferenceAttack2Model(pl.LightningModule):
    def __init__(
        self, 
        encoder_weights,
        angle_dis_weights,
        dims,
        num_classes,
        arch='resnet56',
        resnet_variant='alpha',
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
        super(InferenceAttack2Model, self).__init__()
        self.save_hyperparameters()

        # Load pretrained models
        # Complex encoder
        self.encoder = ComplexEncoder(encoder_weights, drop_imag=False)
        self.encoder.freeze()

        # Angle discriminator
        in_size = get_encoder_output_size(self.encoder, dims)
        self.angle_discriminator = AngleDiscriminator(angle_dis_weights, in_size)
        self.angle_discriminator.freeze()

        self.complex_to_real = ComplexToReal()

        # Load attack classification model
        _, self.processor, self.decoder = load_baseline_network(
            arch,
            num_classes,
            resnet_variant,
            additional_layers=False
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        print(self.hparams)

    def forward(self, x):
        # Reconstruct encoder features
        with torch.no_grad():
            feats, _ = self.encoder(x) # Encoder features
            pred_theta = self.angle_discriminator(feats) # Predict angle
            a_pred = self.complex_to_real(feats, pred_theta) # Revert to real using predicted angle        

        # Classify on unrotated encoder features
        h = self.processor(a_pred)
        out = self.decoder(h)

        return out

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

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.val_acc(pred, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

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
        parser.add_argument( '--angle_dis_weights', type=Path, help='Path to angle discriminator checkpoint.', required=True)
        parser.add_argument( '--encoder_weights', type=Path, help='Path to complex encoder checkpoint.', required=True)
        
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

        parser.add_argument('--arch', type=str, default='resnet56',
            help='Network architecture (resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | lenet | alexnet | vgg)', )
        parser.add_argument('--resnet_variant', type=str, help='alpha | beta', default='alpha')
        
        return parser

class InferenceAttack3Model(pl.LightningModule):
    def __init__(
        self, 
        encoder_weights,
        angle_dis_weights,
        inversion_net_weights,
        dims,
        num_classes,
        arch='resnet56',
        resnet_variant='alpha',
        additional_layers=False,
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
        super(InferenceAttack3Model, self).__init__()
        self.save_hyperparameters()

        # Load pretrained models
        # Complex encoder
        self.protype_encoder = ComplexEncoder(encoder_weights, drop_imag=False)
        self.protype_encoder.freeze()

        # Feature inversion network
        in_size = get_encoder_output_size(self.protype_encoder, dims)
        self.inversion_network = InversionNetwork(inversion_net_weights, in_size[1])
        self.inversion_network.freeze()
        
        # Angle discriminator
        self.angle_discriminator = AngleDiscriminator(angle_dis_weights, in_size)
        self.angle_discriminator.freeze()

        self.complex_to_real = ComplexToReal()

        # Load attack classification model
        self.encoder, self.processor, self.decoder = load_baseline_network(
            arch,
            num_classes,
            resnet_variant,
            additional_layers=False
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        print(self.hparams)

    def forward(self, x):
        # Reconstruct encoder features
        with torch.no_grad():
            feats, _ = self.protype_encoder(x) # Encoder features
            pred_theta = self.angle_discriminator(feats) # Predict angle
            a_pred = self.complex_to_real(feats, pred_theta) # Revert to real using predicted angle        
            x_pred = self.inversion_network(a_pred) # Reconstruct x

        # Classify on unrotated encoder features
        a = self.encoder(x_pred)
        h = self.processor(a)
        out = self.decoder(h)

        return out

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

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.val_acc(pred, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

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
        parser.add_argument( '--angle_dis_weights', type=Path, help='Path to angle discriminator checkpoint.', required=True)
        parser.add_argument( '--encoder_weights', type=Path, help='Path to complex encoder checkpoint.', required=True)
        parser.add_argument( '--inversion_net_weights', type=Path, help='Path to feature inversion network checkpoint.', required=True)
        
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

        parser.add_argument('--arch', type=str, default='resnet56',
            help='Network architecture (resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | lenet | alexnet | vgg)', )
        parser.add_argument('--resnet_variant', type=str, help='alpha | beta', default='alpha')
        parser.add_argument('--additional_layers', action='store_true', help='Add additional layers to network.')
        
        return parser

