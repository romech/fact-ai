import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
from argparse import ArgumentParser

from src.model_inversion_attack import AngleInversionModel, FeatureInversionModel, FeatureInversionAngleModel
from src.dataset import get_datamodule

# Load arguments
parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)

# Which attack model
parser.add_argument('--attack', type=str, required=True,
    help='Name of attack (angle_inversion | feature_inversion | feature_inversion_angle).')
temp_args, _ = parser.parse_known_args()

# Parse args for selected attack
parser.add_argument('--test', action='store_true', help='Perform model evaluation .')
if temp_args.attack == 'angle_inversion':
    parser = AngleInversionModel.add_model_specific_args(parser)
elif temp_args.attack == 'feature_inversion':
    parser = FeatureInversionModel.add_model_specific_args(parser)
elif temp_args.attack == 'feature_inversion_angle':
    parser = FeatureInversionAngleModel.add_model_specific_args(parser)
else:
    raise NotImplementedError(f'{temp_args.attack} is not an available attack.')

args = parser.parse_args()

pl.seed_everything(args.seed)

# Define callbacks
tb_logger = TensorBoardLogger(
    save_dir=args.output_path,
    name=args.experiment_name
)

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(tb_logger.root_dir, 'best-{epoch}-{val_mae:.4f}'),
    save_top_k=1,
    monitor='val_mae',
    mode='min',
    save_last=True,
)

# Load datamodule
dm = get_datamodule(args)
args.num_classes = dm.num_classes
args.dims = dm.dims

# Load model
if args.attack == 'angle_inversion':
    model = AngleInversionModel(
        encoder_weights=args.encoder_weights,
        lr=args.lr,
        schedule=args.schedule,
        steps=args.steps,
        step_factor=args.step_factor,
        dims=args.dims
    )
elif args.attack == 'feature_inversion':    
    model = FeatureInversionModel(
        encoder_weights=args.encoder_weights,
        complex=args.complex,
        lr=args.lr,
        schedule=args.schedule,
        steps=args.steps,
        step_factor=args.step_factor,
        dims=args.dims
    )
elif args.attack == 'feature_inversion_angle':    
    model = FeatureInversionAngleModel(
        encoder_weights=args.encoder_weights,
        angle_dis_weights=args.angle_dis_weights,
        lr=args.lr,
        schedule=args.schedule,
        steps=args.steps,
        step_factor=args.step_factor,
        dims=args.dims
    )


# Run trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    checkpoint_callback=checkpoint_callback,
    logger=tb_logger,
)
trainer.logger._default_hp_metric = None

if not args.test:
    trainer.tune(model, dm)
    trainer.fit(model, dm)
else:
    trainer.test(model, datamodule=dm)
