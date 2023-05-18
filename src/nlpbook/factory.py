from pathlib import Path

import pytorch_lightning as pl
import torch
from flask import Flask, request, jsonify, render_template
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from chrisbase.io import merge_dicts
from chrisbase.time import now
from nlpbook.arguments import TrainerArguments, TesterArguments, ServerArguments


def setup_csv_out(args: ServerArguments, version=None) -> ServerArguments:
    if not version:
        version = now(f'{args.tag}-%m%d.%H%M')
    csv_out: CSVLogger = CSVLogger(args.model.finetuning_home, args.data.name, version)
    args.output.dir_path = Path(csv_out.log_dir)
    args.output.csv_out = csv_out
    return args


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


def make_trainer(args: TrainerArguments) -> pl.Trainer:
    logging_callback = LoggingCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output.dir_path,
        filename=args.model.finetuning_name,
        save_top_k=args.learning.num_save,
        monitor=args.learning.condition.split()[1],
        mode=args.learning.condition.split()[0],
    )
    trainer = pl.Trainer(
        logger=args.output.csv_out,
        devices=args.hardware.devices if not args.hardware.devices else None,
        strategy=args.hardware.strategy if not args.hardware.strategy else None,
        precision=args.hardware.precision if args.hardware.precision else 32,
        accelerator=args.hardware.accelerator if args.hardware.accelerator else None,
        deterministic=torch.cuda.is_available() and args.learning.seed is not None,
        num_sanity_val_steps=0,
        val_check_interval=args.learning.log_steps,
        log_every_n_steps=args.learning.log_steps,
        max_epochs=args.learning.epochs,
        callbacks=[logging_callback, checkpoint_callback],
    )
    return trainer


def make_tester(args: TesterArguments) -> pl.Trainer:
    tester = pl.Trainer(
        logger=args.output.csv_out,
        devices=args.hardware.devices if not args.hardware.devices else None,
        strategy=args.hardware.strategy if not args.hardware.strategy else None,
        precision=args.hardware.precision if args.hardware.precision else 32,
        accelerator=args.hardware.accelerator if args.hardware.accelerator else None,
    )
    return tester


def make_server(inference_fn, template_file, ngrok_home=None):
    app = Flask(__name__, template_folder='')
    if ngrok_home:
        from flask_ngrok import run_with_ngrok
        run_with_ngrok(app, home=ngrok_home)
    else:
        from flask_cors import CORS
        CORS(app)

    @app.route('/')
    def index():
        return render_template(template_file)

    @app.route('/api', methods=['POST'])
    def api():
        query_sentence = request.json
        output_data = inference_fn(query_sentence)
        response = jsonify(output_data)
        return response

    return app
