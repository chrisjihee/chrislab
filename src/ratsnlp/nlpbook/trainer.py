import os

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from ratsnlp.nlpbook.arguments import NLUTrainerArguments


def get_trainer(args: NLUTrainerArguments, return_trainer_only=True):
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
        deterministic=torch.cuda.is_available() and args.seed is not None,
        accelerator=args.accelerator if args.accelerator else None,
        precision=args.precision if args.precision else 32,
        strategy=args.strategy if not args.strategy else None,
        devices=args.devices if not args.devices else None,
    )
    if return_trainer_only:
        return trainer
    else:
        return checkpoint_callback, trainer
