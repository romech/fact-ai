import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from argparse import ArgumentParser
from pathlib import Path
from src.networks.complex import ComplexToReal
from src.utils import get_encoder_output_size
from src.frozen_models import *


class AngleInversionModel(pl.LightningModule):
    def __init__(
            self,
            encoder_weights,
            dims,
            lr=0.001,
            schedule='none',
            steps=[10],
            step_factor=0.1
    ):
        super(AngleInversionModel, self).__init__()
        self.save_hyperparameters()

        # Load pretrained encoder
        self.prototype = ComplexEncoder(encoder_weights)
        self.prototype.freeze()

        # Initialize attack models
        in_size = get_encoder_output_size(self.prototype, dims)
        in_size[1] *= 2
        self.angle_discriminator = Discriminator(in_size[1:])

        print(self.hparams)

    def forward(self, x):
        feats, theta = self.prototype(x)  # Encoder features
        pred_theta = self.angle_discriminator(
            torch.cat([feats[:, 0], feats[:, 1]], dim=1)).squeeze(1)  # Predict angle
        return pred_theta, theta

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred_theta, theta = self(x)

        # Angle prediction loss and error
        loss = F.l1_loss(pred_theta, theta)

        # Log
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred_theta, theta = self(x)

        # Angle prediction loss and error
        loss = F.l1_loss(pred_theta, theta)

        # Log
        self.log('val_loss', loss)

        return OrderedDict({
            'val_loss': loss,
        })

    def test_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through model
        pred_theta, theta = self(x)

        # Calculate loss and reconstruction error
        mae = F.l1_loss(pred_theta, theta)
        self.log('test_mae', mae)

        return mae

    def validation_epoch_end(self, outputs):
        # Print validation results
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f'Validation Loss: {avg_loss}')

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(
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
    def load_from_checkpoint(path):
        ckpt = torch.load(path)
        model = AngleInversionModel(**ckpt["hyper_parameters"])

        new_state_dict = {}
        for k, v in ckpt["state_dict"].items():
            new_state_dict[k.replace("angle_discriminator.", "")] = v

        model.angle_discriminator.load_state_dict(new_state_dict)

        return model

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
    def __init__(
            self,
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
        self.encoder = ComplexEncoder(encoder_weights)
        self.encoder.freeze()
        self.complex_to_real = ComplexToReal()

        # Load pretrained angle discriminator
        in_size = get_encoder_output_size(self.encoder, dims)
        self.angle_discriminator = AngleDiscriminator(angle_dis_weights, in_size.copy())
        self.angle_discriminator.freeze()

        # Initialize attack models
        self.inversion_network = UNet(
            in_channels=in_size[1],
            out_size=dims[1:]
        )

        print(self.hparams)

    def forward(self, x):
        feats, _ = self.encoder(x)  # Encoder features
        pred_theta = self.angle_discriminator(feats)  # Predict angle
        a_pred = self.complex_to_real(feats, pred_theta)  # Revert to real using predicted angle
        out = self.inversion_network(a_pred)  # Reconstruct x

        return out

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred = self(x)

        # Reconstruction loss and error
        loss = F.l1_loss(pred, x)

        # Log
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred = self(x)

        # Reconstruction loss and error
        loss = F.l1_loss(pred, x)

        # Log
        self.log('val_loss', loss)

        return OrderedDict({
            'val_loss': loss,
            'val_sample': torch.cat([x[:10], pred[:10]], dim=3) if batch_idx == 0 else None
        })

    def test_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and reconstruction error
        mae = F.l1_loss(pred, x)
        self.log('test_mae', mae)

        return mae

    def validation_epoch_end(self, outputs):
        # Visualize reconstruction
        grid = make_grid(outputs[0]['val_sample'], nrow=2, normalize=True)
        tensorboard = self.logger.experiment
        tensorboard.add_image('val_samples', grid, self.current_epoch)

        # Print validation results
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f'Validation Loss: {avg_loss}')

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
    def load_from_checkpoint(path):
        ckpt = torch.load(path)
        model = FeatureInversionAngleModel(**ckpt["hyper_parameters"])

        new_state_dict = {}
        for k, v in ckpt["state_dict"].items():
            new_state_dict[k.replace("inversion_network.", "")] = v

        model.inversion_network.load_state_dict(new_state_dict)

        return model

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
            encoder_weights,
            dims,
            complex=False,
            lr=0.001,
            schedule='none',
            steps=[100, 150],
            step_factor=0.1,
    ):
        super(FeatureInversionModel, self).__init__()
        self.save_hyperparameters()

        if complex:
            self.prototype = ComplexEncoder(encoder_weights)
        else:
            self.prototype = BaselineEncoder(encoder_weights)
        self.prototype.freeze()

        enc_output_size = get_encoder_output_size(self.prototype, dims)
        if complex:
            channels = enc_output_size[1] * 2
        else:
            channels = enc_output_size[0]

        self.inversion_network = UNet(
            in_channels=channels,
            out_size=dims[1:]
        )

        print(self.hparams)

    def forward(self, x):
        if self.hparams.complex:
            feats, _ = self.prototype(x)  # Encoder features
            feats = torch.cat([feats[:, 0], feats[:, 1]], dim=1)
        else:
            feats = self.prototype(x)  # Encoder features

        out = self.inversion_network(feats)  # Reconstruction of x
        return out

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through network
        pred = self(x)

        # Calculate loss and reconstruction error
        loss = F.l1_loss(pred, x)
        self.log('train_loss', loss)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and reconstruction error
        loss = F.l1_loss(pred, x)
        self.log('val_loss', loss)

        return OrderedDict({
            'val_loss': loss,
            'val_sample': torch.cat([x[:10], pred[:10]], dim=3) if batch_idx == 0 else None
        })

    def test_step(self, batch, batch_idx):
        x, _ = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and reconstruction error
        mae = F.l1_loss(pred, x)
        self.log('test_mae', mae)

        return mae

    def validation_epoch_end(self, outputs):
        # Visualize reconstruction
        grid = make_grid(outputs[0]['val_sample'], nrow=2, normalize=True)
        tensorboard = self.logger.experiment
        tensorboard.add_image('val_samples', grid, self.current_epoch)

        # Print validation results
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f'Validation Loss: {avg_loss} ')

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
    def load_from_checkpoint(path):
        ckpt = torch.load(path)
        model = FeatureInversionModel(**ckpt["hyper_parameters"])

        new_state_dict = {}
        for k, v in ckpt["state_dict"].items():
            new_state_dict[k.replace("inversion_network.", "")] = v

        model.inversion_network.load_state_dict(new_state_dict)

        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--experiment_name', type=str, help='Name of experiment.', default='default')
        parser.add_argument('--data_path', type=Path, help='Path to download data.', default='data/')
        parser.add_argument('--output_path', type=Path, help='Path to save output.', default='output/')
        parser.add_argument('--encoder_weights', type=Path, help='Path to pretrained network checkpoint.', required=True)
        parser.add_argument('--complex', action='store_true', help='Loading a complex pretrained network?.')
        parser.add_argument('--seed', type=int, help='Seed to allow for reproducible results.', default=0)

        parser.add_argument('--dataset', type=str, help='cifar10 | cifar100 | celeba | cub200', default='cifar10')
        parser.add_argument('--workers', type=int, help='Number of dataloader workers.', default=6)
        parser.add_argument('--batch_size', type=int, help='Number of per batch samples.', default=64)
        parser.add_argument('--lr', type=float, help='Learning rate.', default=0.001)
        parser.add_argument('--schedule', type=str, help='Learning rate schedule (none | step)', default='none')
        parser.add_argument('--steps', nargs='+', type=int, help='Epochs where LR is reduced in step schedule.', default=[100, 150])
        parser.add_argument('--step_factor', type=float, help='Step reduction rate in step schedule .', default=0.1)
        
        return parser
