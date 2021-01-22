import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
from argparse import ArgumentParser

from src.model_attack import Inversion2Model
from src.dataset import get_datamodule

# Load arguments
parser = ArgumentParser()
parser = Inversion2Model.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
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
model = Inversion2Model(
    weights=args.weights,
    complex=args.complex,
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

trainer.tune(model, dm)
trainer.fit(model, dm)