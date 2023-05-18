import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from dataclasses_json import DataClassJsonMixin
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.strategies import Strategy

from chrisbase.io import ProjectEnv, make_dir
from chrisbase.io import files, make_parent_dir, out_hr, out_table
from chrisbase.time import now, str_delta
from chrisbase.util import to_dataframe


@dataclass
class DataFiles(DataClassJsonMixin):
    train: str | Path | None = field(default=None)
    valid: str | Path | None = field(default=None)
    test: str | Path | None = field(default=None)


@dataclass
class ModelArgs(DataClassJsonMixin):
    data_file: DataFiles | None = field(default=None)
    data_name: str | Path | None = field(default=None)
    data_home: str | Path | None = field(default=None)
    data_caching: bool = field(default=False)
    data_download: bool = field(default=False)  # force to download downstream data and pretrained models
    finetuned_name: str | Path | None = field(default=None)  # filename or filename format of downstream model
    finetuned_home: str | Path | None = field(default=None)
    pretrained_name: str | Path = field(default="bert-base-multilingual-cased")
    max_seq_length: int = field(default=128)  # maximum total input sequence length after tokenization

    def __post_init__(self):
        assert isinstance(self.finetuned_home, (str, Path)), "finetuned_home must be str or Path"
        if self.data_home:
            self.data_home = Path(self.data_home)
        if self.finetuned_home:
            self.finetuned_home = make_dir(self.finetuned_home)


@dataclass
class JobTimer(DataClassJsonMixin):
    t1 = datetime.now()
    t2 = datetime.now()
    started_time: str | None = field(init=False)
    settled_time: str | None = field(init=False)
    elapsed_time: str | None = field(init=False)

    def __post_init__(self):
        self.set_started_time()

    def set_started_time(self):
        self.started_time = now()
        self.settled_time = None
        self.elapsed_time = None
        self.t1 = datetime.now()
        return self

    def set_settled_time(self):
        self.t2 = datetime.now()
        self.settled_time = now()
        self.elapsed_time = str_delta(self.t2 - self.t1)
        return self


@dataclass
class JobResult(DataClassJsonMixin):
    metrics: dict = field(default_factory=dict)


@dataclass
class CommonArguments(DataClassJsonMixin):
    action = "base"
    env: ProjectEnv = field()
    model: ModelArgs = field()
    timer: JobTimer = field(default=JobTimer())
    result: JobResult = field(default=JobResult())

    def __post_init__(self):
        if not self.env.argument_file.stem.endswith(self.action):
            self.env.argument_file = self.env.argument_file.with_stem(f"{self.env.argument_file.stem}-{self.action}")

    def as_dataframe(self, columns=None):
        if not columns:
            columns = [self.__class__.__name__, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.model, data_prefix="model"),
            to_dataframe(columns=columns, raw=self.timer, data_prefix="timer"),
            to_dataframe(columns=columns, raw=self.result, data_prefix="result"),
        ]).reset_index(drop=True)

    def print_dataframe(self):
        out_hr(c='-')
        out_table(self.as_dataframe())
        out_hr(c='-')
        return self

    def save_arguments(self, to: Path | str = None) -> Path:
        args_file = to if to else self.model.finetuned_home / self.model.data_name / self.env.argument_file
        make_parent_dir(args_file).write_text(self.to_json(default=str, ensure_ascii=False, indent=2))
        return args_file


@dataclass
class ServerArguments(CommonArguments):
    action = "serve"

    def __post_init__(self):
        super().__post_init__()
        if self.action in ("serve", "test"):
            assert self.model.finetuned_home.exists() and self.model.finetuned_home.is_dir(), \
                f"No model directory: {self.model.finetuned_home}"
            if not self.model.finetuned_name:
                ckpt_files = files(self.model.finetuned_home / self.model.data_name / "*" / "*.ckpt")  # TODO: NEED TO CHECK
                ckpt_files = sorted([x for x in ckpt_files if "temp" not in str(x) and "tmp" not in str(x)], key=str)
                assert len(ckpt_files) > 0, f"No checkpoint file in {self.model.finetuned_home}"
                self.model.finetuned_name = ckpt_files[-1].name
            assert (self.model.finetuned_home / self.model.finetuned_name).exists(), \
                f"No checkpoint file: {self.model.finetuned_home / self.model.finetuned_name}"


@dataclass
class HardwareArgs(DataClassJsonMixin):
    batch_size: int = field(default=32)
    cpu_workers: int = field(default=os.cpu_count())
    accelerator: str | Accelerator = field(default="auto")  # possbile value: "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"
    precision: int | str = field(default=32)  # floating-point precision type
    strategy: str | Strategy | None = field(default=None)  # multi-device strategies
    devices: List[int] | int | str = field(default=1)  # devices to use

    def __post_init__(self):
        if not self.strategy:
            if self.devices == 1 or isinstance(self.devices, list) and len(self.devices) == 1:
                self.strategy = "single_device"
            elif isinstance(self.devices, int) and self.devices > 1 or isinstance(self.devices, list) and len(self.devices) > 1:
                self.strategy = "ddp"


@dataclass
class TesterArguments(ServerArguments):
    action = "test"
    hardware: HardwareArgs = field(default=HardwareArgs(), metadata={"help": "device information"})

    def as_dataframe(self, columns=None):
        if not columns:
            columns = [self.__class__.__name__, "value"]
        return pd.concat([
            super().as_dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.hardware, data_prefix="hardware"),
        ]).reset_index(drop=True)


@dataclass
class TrainingArgs(DataClassJsonMixin):
    learning_rate: float = field(default=5e-5)
    epochs: int = field(default=1)
    log_steps: int = field(default=10)
    save_top_k: int = field(default=100)
    monitor: str = field(default="min val_loss")  # monitor condition for save_top_k
    seed: int | None = field(default=None)  # random seed


@dataclass
class TrainerArguments(TesterArguments):
    action = "train"
    training: TrainingArgs = field(default=TrainingArgs())

    def as_dataframe(self, columns=None):
        if not columns:
            columns = [self.__class__.__name__, "value"]
        return pd.concat([
            super().as_dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.training, data_prefix="training"),
        ]).reset_index(drop=True)


class UsingArguments:
    def __init__(self, args: CommonArguments):
        self.args: CommonArguments = args

    def __enter__(self) -> Path:
        self.args_file: Path = self.args.save_arguments()
        return self.args_file

    def __exit__(self, *exc_info):
        self.args_file.unlink(missing_ok=True)


class CheckingRuntime:
    def __init__(self, args: CommonArguments, outdir: str | Path):
        self.args: CommonArguments = args
        self.outdir: Path = Path(outdir)

    def __enter__(self):
        self.args.timer.set_started_time()

    def __exit__(self, *exc_info):
        self.args.timer.set_settled_time()
        self.args.save_arguments(self.outdir / self.args.env.argument_file)
