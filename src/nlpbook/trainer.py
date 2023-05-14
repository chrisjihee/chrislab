import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from chrisbase.io import make_dir
from nlpbook.arguments import NLUTrainerArguments, NLUTesterArguments


def get_trainer(args: NLUTrainerArguments):
    downstream_model_home = make_dir(args.downstream_model_home)
    checkpoint_callback = ModelCheckpoint(
        dirpath=downstream_model_home,
        filename=args.downstream_model_file,
        save_top_k=args.save_top_k,
        monitor=args.monitor.split()[1],
        mode=args.monitor.split()[0],
    )
    trainer = Trainer(
        num_sanity_val_steps=0,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=downstream_model_home,
        deterministic=torch.cuda.is_available() and args.seed is not None,
        accelerator=args.accelerator if args.accelerator else None,
        precision=args.precision if args.precision else 32,
        strategy=args.strategy if not args.strategy else None,
        devices=args.devices if not args.devices else None,
    )
    return trainer


def get_tester(args: NLUTesterArguments):
    trainer = Trainer(
        fast_dev_run=True,
        accelerator=args.accelerator if args.accelerator else None,
        precision=args.precision if args.precision else 32,
        strategy=args.strategy if not args.strategy else None,
        devices=args.devices if not args.devices else None,
    )
    return trainer
