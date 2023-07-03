import lightning.pytorch as pl
import torch
from flask import Flask, request, jsonify, render_template
from lightning.pytorch.callbacks import ModelCheckpoint

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


def make_trainer(args: TrainerArguments) -> pl.Trainer:
    logging_callback = LoggingCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.env.output_home,
        filename=args.model.finetuning_name,
        save_top_k=args.learning.num_keeping,
        monitor=args.learning.keeping_by.split()[1],
        mode=args.learning.keeping_by.split()[0],
    )
    trainer = pl.Trainer(
        logger=args.env.csv_logger,
        devices=args.hardware.devices,
        strategy=args.hardware.strategy,
        precision=args.hardware.precision,
        accelerator=args.hardware.accelerator,
        deterministic=torch.cuda.is_available() and args.learning.seed is not None,
        # enable_progress_bar=False,
        num_sanity_val_steps=0,
        val_check_interval=args.learning.validating_on,
        max_epochs=args.learning.epochs,
        callbacks=[logging_callback, checkpoint_callback],
    )
    return trainer


def make_tester(args: TesterArguments) -> pl.Trainer:
    tester = pl.Trainer(
        logger=args.env.csv_logger,
        devices=args.hardware.devices,
        strategy=args.hardware.strategy,
        precision=args.hardware.precision,
        accelerator=args.hardware.accelerator,
        # enable_progress_bar=False,
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
