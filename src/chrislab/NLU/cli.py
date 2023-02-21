import torch
from typer import Typer

from chrisbase.io import MyTimer
from chrislab.NLU.finetuner import MyFinetuner
from chrislab.NLU.predictor import MyPredictor

app = Typer()


@app.command()
def check(config: str, prefix: str = "", postfix: str = ""):
    with MyTimer(verbose=True, mute_logger=["lightning.fabric.utilities.seed"]):
        MyFinetuner(config=config, prefix=prefix, postfix=postfix).ready()


@app.command()
def train(config: str, prefix: str = "", postfix: str = ""):
    torch.set_float32_matmul_precision('high')
    with MyTimer(verbose=True, mute_logger=["lightning.fabric.utilities.seed"]):
        MyFinetuner(config=config, prefix=prefix, postfix=postfix).run()


@app.command()
def apply(config: str, prefix: str = "", postfix: str = ""):
    with MyTimer(verbose=True, mute_logger=["lightning.fabric.utilities.seed"]):
        MyPredictor(config=config, prefix=prefix, postfix=postfix).run()
