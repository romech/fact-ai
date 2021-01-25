import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
from argparse import ArgumentParser

from src.model_inference_attack import InferenceAttack1Model, InferenceAttack2Model, InferenceAttack3Model
from src.dataset import get_datamodule

# Load arguments
parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)

# Which attack model
parser.add_argument('--attack', type=str, required=True,
    help='Name of attack (inference1 | inference2).')
temp_args, _ = parser.parse_known_args()

# Parse args for selected attack
parser.add_argument('--test', action='store_true', help='Perform model evaluation .')
if temp_args.attack == 'inference1':
    parser = InferenceAttack1Model.add_model_specific_args(parser)
elif temp_args.attack == 'inference2':
    parser = InferenceAttack2Model.add_model_specific_args(parser)
elif temp_args.attack == 'inference3':
    parser = InferenceAttack3Model.add_model_specific_args(parser)
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

# Load model
if args.attack == 'inference1':
    model = InferenceAttack1Model(
        classifier_weights=args.classifier_weights,
        encoder_weights=args.encoder_weights,
        inversion_net_weights=args.inversion_net_weights,
        angle_dis_weights=args.angle_dis_weights,
        dims=args.dims,
        complex=args.complex
    )
elif args.attack == 'inference2':
    model = InferenceAttack2Model(
        encoder_weights=args.encoder_weights,
        angle_dis_weights=args.angle_dis_weights,
        dims=args.dims,
        num_classes=args.num_classes,
        arch=args.arch,
        resnet_variant=args.resnet_variant,
        optimizer=args.optimizer,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        schedule=args.schedule,
        steps=args.steps,
        step_factor=args.step_factor,
        complex=args.complex
    )
elif args.attack == 'inference3':
    model = InferenceAttack3Model(
        encoder_weights=args.encoder_weights,
        angle_dis_weights=args.angle_dis_weights,
        inversion_net_weights=args.inversion_net_weights,
        dims=args.dims,
        num_classes=args.num_classes,
        arch=args.arch,
        resnet_variant=args.resnet_variant,
        additional_layers=args.additional_layers,
        optimizer=args.optimizer,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        schedule=args.schedule,
        steps=args.steps,
        step_factor=args.step_factor,
        complex=args.complex
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