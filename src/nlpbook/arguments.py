import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from dataclasses_json import DataClassJsonMixin

from chrisbase.io import ProjectEnv, make_dir
from chrisbase.io import files, make_parent_dir, out_hr, out_table
from chrisbase.time import now, str_delta
from chrisbase.util import to_dataframe
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import Strategy


@dataclass
class TypedData(DataClassJsonMixin):
    data_type = None

    def __post_init__(self):
        self.data_type = self.__class__.__name__


@dataclass
class OptionData(TypedData):
    def __post_init__(self):
        super().__post_init__()


@dataclass
class ResultData(TypedData):
    def __post_init__(self):
        super().__post_init__()


@dataclass
class ArgumentGroupData(TypedData):
    tag = None

    def __post_init__(self):
        super().__post_init__()


@dataclass
class DataFiles(DataClassJsonMixin):
    train: str | Path | None = field(default=None)
    valid: str | Path | None = field(default=None)
    test: str | Path | None = field(default=None)


@dataclass
class DataOption(OptionData):
    name: str | Path = field()
    home: str | Path | None = field(default=None)
    files: DataFiles | None = field(default=None)
    caching: bool = field(default=False)
    redownload: bool = field(default=False)
    show_examples: int = field(default=3)

    def __post_init__(self):
        if self.home:
            self.home = Path(self.home)


@dataclass
class ModelOption(OptionData):
    pretrained: str | Path = field()
    finetuning_home: str | Path = field()
    finetuning_name: str | Path | None = field(default=None)  # filename or filename format of downstream model
    max_seq_length: int = field(default=128)  # maximum total input sequence length after tokenization

    def __post_init__(self):
        self.finetuning_home = Path(self.finetuning_home)


@dataclass
class HardwareOption(OptionData):
    cpu_workers: int = field(default=os.cpu_count())
    accelerator: str | Accelerator = field(default="auto")  # possbile value: "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"
    batch_size: int = field(default=32)
    precision: int | str = field(default=32)  # floating-point precision type
    strategy: str | Strategy = field(default="auto")  # multi-device strategies
    devices: List[int] | int | str = field(default="auto")  # devices to use

    def __post_init__(self):
        if not self.strategy:
            if self.devices == 1 or isinstance(self.devices, list) and len(self.devices) == 1:
                self.strategy = "single_device"
            elif isinstance(self.devices, int) and self.devices > 1 or isinstance(self.devices, list) and len(self.devices) > 1:
                self.strategy = "ddp"


@dataclass
class LearningOption(OptionData):
    validating_fmt: str | None = field(default=None)
    validating_on: int | float = field(default=1.0)
    num_keeping: int = field(default=5)
    keeping_by: str = field(default="min val_loss")
    epochs: int = field(default=1)
    speed: float = field(default=5e-5)
    seed: int | None = field(default=None)  # random seed

    def __post_init__(self):
        self.validating_on = math.fabs(self.validating_on)


@dataclass
class JobTimer(ResultData):
    name: str = field()
    t1 = datetime.now()
    t2 = datetime.now()
    started: str | None = field(init=False)
    settled: str | None = field(init=False)
    elapsed: str | None = field(init=False)

    def __post_init__(self):
        self.set_started()

    def set_started(self):
        self.started = now()
        self.settled = None
        self.elapsed = None
        self.t1 = datetime.now()
        return self

    def set_settled(self):
        self.t2 = datetime.now()
        self.settled = now()
        self.elapsed = str_delta(self.t2 - self.t1)
        return self


@dataclass
class JobOutput(ResultData):
    dir_path: Path | None = field(init=False, default=None)
    csv_out: CSVLogger | None = field(init=False, default=None)
    result: dict = field(init=False, default_factory=dict)
    global_step: int = field(init=False, default=0)
    global_epoch: float = field(init=False, default=0.0)
    epoch_per_step: float = field(init=False, default=0.0)


@dataclass
class CommonArguments(ArgumentGroupData):
    tag = "common"
    job: JobTimer = field()
    env: ProjectEnv = field()
    data: DataOption | None = field(default=None)
    model: ModelOption | None = field(default=None)
    output: JobOutput = field(default=JobOutput())

    def __post_init__(self):
        super().__post_init__()
        if not self.env.argument_file.stem.endswith(self.tag):
            self.env.argument_file = self.env.argument_file.with_stem(f"{self.env.argument_file.stem}-{self.tag}")
        if self.data and self.model:
            self.output.dir_path = self.model.finetuning_home / self.data.name
        if self.model and self.tag in ("train",):
            self.model.finetuning_home = make_dir(self.model.finetuning_home)

    def dataframe(self, columns=None):
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.job, data_prefix="job"),
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.model, data_prefix="model"),
            to_dataframe(columns=columns, raw=self.output, data_prefix="output"),
        ]).reset_index(drop=True)

    def show(self):
        out_hr(c='-')
        out_table(self.dataframe())
        out_hr(c='-')
        return self

    def setup_csv_out(self, version=None):
        if not version:
            version = now('%m%d.%H%M%S')
        self.output.csv_out = CSVLogger(root_dir=self.model.finetuning_home, name=self.data.name,
                                        version=f'{self.tag}-{self.job.name}-{version}',
                                        flush_logs_every_n_steps=1)
        self.output.dir_path = Path(self.output.csv_out.log_dir)
        return self

    def save(self, to: Path | str = None) -> Path | None:
        if not self.output.dir_path:
            return None
        args_file = to if to else self.output.dir_path / self.env.argument_file
        args_json = self.to_json(default=str, ensure_ascii=False, indent=2)
        make_parent_dir(args_file).write_text(args_json, encoding="utf-8")
        return args_file


@dataclass
class ServerArguments(CommonArguments):
    tag = "serve"

    def __post_init__(self):
        super().__post_init__()
        if self.tag in ("serve", "test"):
            assert self.model.finetuning_home.exists() and self.model.finetuning_home.is_dir(), \
                f"No finetuning home: {self.model.finetuning_home}"
            if not self.model.finetuning_name:
                ckpt_files: List[Path] = files(self.output.dir_path / "**/*.ckpt")
                assert ckpt_files, f"No checkpoint file in {self.model.finetuning_home}"
                ckpt_files = sorted([x for x in ckpt_files if "temp" not in str(x) and "tmp" not in str(x)], key=str)
                self.model.finetuning_name = ckpt_files[-1].relative_to(self.output.dir_path)
            assert (self.output.dir_path / self.model.finetuning_name).exists() and (self.output.dir_path / self.model.finetuning_name).is_file(), \
                f"No checkpoint file: {self.output.dir_path / self.model.finetuning_name}"


@dataclass
class TesterArguments(ServerArguments):
    tag = "test"
    hardware: HardwareOption = field(default=HardwareOption(), metadata={"help": "device information"})

    def dataframe(self, columns=None):
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.hardware, data_prefix="hardware"),
        ]).reset_index(drop=True)


@dataclass
class TrainerArguments(TesterArguments):
    tag = "train"
    learning: LearningOption = field(default=LearningOption())

    def dataframe(self, columns=None):
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.learning, data_prefix="training"),
        ]).reset_index(drop=True)


class ArgumentsUsing:
    def __init__(self, args: CommonArguments):
        self.args: CommonArguments = args

    def __enter__(self) -> Path:
        self.args_file: Path | None = self.args.save()
        return self.args_file

    def __exit__(self, *exc_info):
        if self.args_file:
            self.args_file.unlink(missing_ok=True)


class RuntimeChecking:
    def __init__(self, args: CommonArguments):
        self.args: CommonArguments = args

    def __enter__(self):
        self.args.job.set_started()

    def __exit__(self, *exc_info):
        self.args.job.set_settled()
        self.args.save(self.args.output.dir_path / self.args.env.argument_file)
