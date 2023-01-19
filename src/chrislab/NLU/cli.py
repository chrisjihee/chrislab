from typer import Typer

from .predictor import *

app = Typer()


@app.command()
def preparing(config: str, prefix: str = "", postfix: str = ""):
    with MyTimer(verbose=True, mute_logger=["pytorch_lightning.utilities.seed"]):
        MyFinetuner(config=config, prefix=prefix, postfix=postfix).ready()


@app.command()
def finetuning(config: str, prefix: str = "", postfix: str = ""):
    with MyTimer(verbose=True, mute_logger=["pytorch_lightning.utilities.seed", "pytorch_lightning.utilities.distributed"]):
        MyFinetuner(config=config, prefix=prefix, postfix=postfix).run()


@app.command()
def predicting(config: str, prefix: str = "", postfix: str = ""):
    with MyTimer(verbose=True, mute_logger=["pytorch_lightning.utilities.seed", "pytorch_lightning.utilities.distributed"]):
        MyPredictor(config=config, prefix=prefix, postfix=postfix).run()
