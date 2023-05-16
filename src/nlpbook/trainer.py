from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from chrisbase.io import make_dir, merge_dicts
from nlpbook.arguments import NLUTrainerArguments, NLUTesterArguments


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics = merge_dicts(
            {
                "step": 0,
                "current_epoch": pl_module.current_epoch,
                "global_rank": pl_module.global_rank,
                "global_step": pl_module.global_step,
                "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
            },
            trainer.callback_metrics,
        )
        pl_module.logger.log_metrics(metrics)


def get_trainer(args: NLUTrainerArguments) -> pl.Trainer:
    train_logger = CSVLogger(args.downstream_model_home, name=args.downstream_data_name)
    logging_callback = LoggingCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(train_logger.log_dir),
        filename=args.downstream_model_file,
        save_top_k=args.save_top_k,
        monitor=args.monitor.split()[1],
        mode=args.monitor.split()[0],
    )
    trainer = pl.Trainer(
        log_every_n_steps=args.log_steps,
        val_check_interval=args.log_steps,
        num_sanity_val_steps=0,
        logger=train_logger,
        callbacks=[logging_callback, checkpoint_callback],
        max_epochs=args.epochs,
        deterministic=torch.cuda.is_available() and args.seed is not None,
        accelerator=args.accelerator if args.accelerator else None,
        precision=args.precision if args.precision else 32,
        strategy=args.strategy if not args.strategy else None,
        devices=args.devices if not args.devices else None,
    )
    return trainer


def get_tester(args: NLUTesterArguments) -> pl.Trainer:
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator=args.accelerator if args.accelerator else None,
        precision=args.precision if args.precision else 32,
        strategy=args.strategy if not args.strategy else None,
        devices=args.devices if not args.devices else None,
    )
    return trainer
