import argparse

from .predictor import *


def command(args):
    mode = str(args.mode).lower()
    if mode == 'preparing':
        with MyTimer(verbose=True, mute_logger=["pytorch_lightning.utilities.seed"]):
            MyFinetuner(config=args.config, prefix=args.prefix, postfix=args.postfix).ready()
    elif mode == 'finetuning':
        with MyTimer(verbose=True, mute_logger=["pytorch_lightning.utilities.seed", "pytorch_lightning.utilities.distributed"]):
            MyFinetuner(config=args.config, prefix=args.prefix, postfix=args.postfix).run()
    elif mode == 'predicting':
        with MyTimer(verbose=True, mute_logger=["pytorch_lightning.utilities.seed", "pytorch_lightning.utilities.distributed"]):
            MyPredictor(config=args.config, prefix=args.prefix, postfix=args.postfix).run()
    else:
        raise ValueError(f"Not available mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model or predict by it for NLU task")
    parser.add_argument("--mode", type=str, required=True, help="run mode [finetuning, predicting]")
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--prefix", type=str, required=False, default="", help="prefix of run")
    parser.add_argument("--postfix", type=str, required=False, default="", help="postfix of run")
    command(parser.parse_args())
