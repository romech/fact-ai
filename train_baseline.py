import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
from argparse import ArgumentParser

from src.model_classifier import BaselineModel
from src.dataset import get_datamodule

# Load arguments
parser = ArgumentParser()
parser = BaselineModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

pl.seed_everything(args.seed)

# Define callbacks
tb_logger = TensorBoardLogger(
    save_dir=args.output_path,
    name=args.experiment_name
)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(tb_logger.root_dir, 'best-{epoch}-{val_acc:.4f}'),
    save_top_k=1,
    monitor='val_acc',
    mode='max',
    save_last=True,
)

# Load datamodule
dm = get_datamodule(args)
args.num_classes = dm.num_classes
args.dims = dm.dims

pl.seed_everything(args.seed)

# Load model
model = BaselineModel(
    arch=args.arch,
    num_classes=args.num_classes,
    additional_layers=args.additional_layers,
    resnet_variant=args.resnet_variant,
    noisy=args.noisy,
    gamma=args.gamma,
    optimizer=args.optimizer,
    lr=args.lr,
    beta1=args.beta1,
    beta2=args.beta2,
    weight_decay=args.weight_decay,
    momentum=args.momentum,
    schedule=args.schedule,
    steps=args.steps,
    step_factor=args.step_factor
)

# Run trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    checkpoint_callback=checkpoint_callback,
    logger=[tb_logger],
)
trainer.logger._default_hp_metric = None

trainer.tune(model, dm)
trainer.fit(model, dm)