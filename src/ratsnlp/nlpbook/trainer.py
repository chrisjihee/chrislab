import os

import torch

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from ratsnlp.nlpbook.classification.arguments import ClassificationTrainArguments


def get_trainer(args: ClassificationTrainArguments, return_trainer_only=True):
    ckpt_path = os.path.abspath(args.downstream_model_home)
    os.makedirs(ckpt_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=args.downstream_model_file,
        save_top_k=args.save_top_k,
        monitor=args.monitor.split()[1],
        mode=args.monitor.split()[0],
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        callbacks=[checkpoint_callback],
        default_root_dir=ckpt_path,
        # For GPU Setup
        deterministic=torch.cuda.is_available() and args.seed is not None,
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32,
    )
    if return_trainer_only:
        return trainer
    else:
        return checkpoint_callback, trainer
