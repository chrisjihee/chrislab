import torch
from flask import Flask, request, jsonify, render_template
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from nlpbook.arguments import TrainerArguments, TesterArguments


def make_trainer(args: TrainerArguments) -> Trainer:
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.env.output_home,
        filename=args.model.name,
        save_top_k=args.learning.num_saving,
        monitor=args.learning.saving_mode.split()[1],
        mode=args.learning.saving_mode.split()[0],
    )
    trainer = Trainer(
        logger=args.prog.csv_logger,
        devices=args.hardware.devices,
        strategy=args.hardware.strategy,
        precision=args.hardware.precision,
        accelerator=args.hardware.accelerator,
        deterministic=torch.cuda.is_available() and args.learning.random_seed is not None,
        log_every_n_steps=1,
        # enable_progress_bar=False,
        num_sanity_val_steps=0,
        val_check_interval=args.learning.check_rate_on_training,
        max_epochs=args.learning.num_epochs,
        callbacks=[checkpoint_callback],
    )
    return trainer


def make_tester(args: TesterArguments) -> Trainer:
    tester = Trainer(
        logger=args.prog.csv_logger,
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
