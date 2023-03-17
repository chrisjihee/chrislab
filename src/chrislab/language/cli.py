import torch
from typer import Typer

from chrisbase.io import JobTimer
from .finetuner import MyFinetuner
from .predictor import MyPredictor

app = Typer()


@app.command()
def check(config: str, prefix: str = "", postfix: str = "", save_cache=True, reset_cache=False, show_state=False, draw_figure=False):
    with JobTimer(verbose=True, mute_logger=["lightning.fabric.utilities.seed"]):
        MyFinetuner(config=config, prefix=prefix, postfix=postfix, save_cache=save_cache, reset_cache=reset_cache).ready(show_state=show_state, draw_figure=draw_figure)


@app.command()
def train(config: str, prefix: str = "", postfix: str = "", save_cache=True, reset_cache=False, show_state=True):
    torch.set_float32_matmul_precision('high')
    with JobTimer(verbose=True, mute_logger=["lightning.fabric.utilities.seed"]):
        MyFinetuner(config=config, prefix=prefix, postfix=postfix, save_cache=save_cache, reset_cache=reset_cache).run(show_state=show_state)


@app.command()
def apply(config: str, prefix: str = "", postfix: str = "", save_cache=True, reset_cache=False, show_state=True):
    with JobTimer(verbose=True, mute_logger=["lightning.fabric.utilities.seed"]):
        MyPredictor(config=config, prefix=prefix, postfix=postfix, save_cache=save_cache, reset_cache=reset_cache).run(show_state=show_state)
