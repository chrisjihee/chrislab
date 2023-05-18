from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from chrisbase.io import merge_dicts
from nlpbook.arguments import TrainerArguments, TesterArguments


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


def get_trainer(args: TrainerArguments) -> pl.Trainer:
    train_logger = CSVLogger(args.model.finetuned_home, name=args.model.data_name)
    logging_callback = LoggingCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(train_logger.log_dir),
        filename=args.model.finetuned_name,
        save_top_k=args.training.save_top_k,
        monitor=args.training.monitor.split()[1],
        mode=args.training.monitor.split()[0],
    )
    trainer = pl.Trainer(
        log_every_n_steps=args.training.log_steps,
        val_check_interval=args.training.log_steps,
        num_sanity_val_steps=0,
        logger=train_logger,
        callbacks=[logging_callback, checkpoint_callback],
        max_epochs=args.training.epochs,
        deterministic=torch.cuda.is_available() and args.training.seed is not None,
        accelerator=args.hardware.accelerator if args.hardware.accelerator else None,
        precision=args.hardware.precision if args.hardware.precision else 32,
        strategy=args.hardware.strategy if not args.hardware.strategy else None,
        devices=args.hardware.devices if not args.hardware.devices else None,
    )
    return trainer


def get_tester(args: TesterArguments) -> pl.Trainer:
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator=args.hardware.accelerator if args.hardware.accelerator else None,
        precision=args.hardware.precision if args.hardware.precision else 32,
        strategy=args.hardware.strategy if not args.hardware.strategy else None,
        devices=args.hardware.devices if not args.hardware.devices else None,
    )
    return trainer
