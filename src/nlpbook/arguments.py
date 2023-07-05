import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from dataclasses_json import DataClassJsonMixin

from chrisbase.io import files, make_parent_dir, hr, str_table, out_hr, out_table, get_hostname, get_hostaddr, running_file, first_or, cwd, configure_dual_logger, configure_unit_logger
from chrisbase.time import now, str_delta
from chrisbase.util import to_dataframe
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.strategies import Strategy
from lightning.pytorch.loggers import CSVLogger

logger = logging.getLogger(__name__)


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
    # cpu_workers: int = field(default=0)
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
class TimeChecker(ResultData):
    t1 = datetime.now()
    t2 = datetime.now()
    started: str | None = field(default=None)
    settled: str | None = field(default=None)
    elapsed: str | None = field(default=None)

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
class ProgressChecker(ResultData):
    result: dict = field(init=False, default_factory=dict)
    global_step: int = field(init=False, default=0)
    global_epoch: float = field(init=False, default=0.0)
    epoch_per_step: float = field(init=False, default=0.0)


@dataclass
class ProjectEnv(TypedData):
    project: str = field()
    job_name: str = field(default=None)
    hostname: str = field(init=False)
    hostaddr: str = field(init=False)
    python_path: Path = field(init=False)
    working_path: Path = field(init=False)
    running_file: Path = field(init=False)
    command_args: List[str] = field(init=False)
    output_home: str | Path | None = field(default=None)
    logging_file: str | Path = field(default="message.out")
    argument_file: str | Path = field(default="arguments.json")
    msg_level: int = field(default=logging.INFO)
    msg_format: str = field(default="%(asctime)s %(levelname)s %(message)s")
    date_format: str = field(default="[%m.%d %H:%M:%S]")
    csv_logger: CSVLogger | None = field(init=False, default=None)

    def set(self, name: str = None):
        self.job_name = name
        return self

    def __post_init__(self):
        assert self.project, "Project name must be provided"
        self.hostname = get_hostname()
        self.hostaddr = get_hostaddr()
        self.python_path = Path(sys.executable)
        self.running_file = running_file()
        self.project_path = first_or([x for x in self.running_file.parents if x.name.startswith(self.project)])
        assert self.project_path, f"Could not find project path for {self.project} in {', '.join([str(x) for x in self.running_file.parents])}"
        self.working_path = cwd(self.project_path)
        self.running_file = self.running_file.relative_to(self.working_path)
        self.command_args = sys.argv[1:]
        self.logging_file = Path(self.logging_file)
        self.argument_file = Path(self.argument_file)
        if self.output_home:
            self.output_home = Path(self.output_home)
            configure_dual_logger(level=self.msg_level, fmt=self.msg_format, datefmt=self.date_format,
                                  filename=self.output_home / self.logging_file)
        else:
            configure_unit_logger(level=self.msg_level, fmt=self.msg_format, datefmt=self.date_format,
                                  stream=sys.stdout)


@dataclass
class CommonArguments(ArgumentGroupData):
    tag = "common"
    env: ProjectEnv = field()
    time: TimeChecker = field(default=TimeChecker())
    prog: ProgressChecker = field(default=ProgressChecker())
    data: DataOption | None = field(default=None)
    model: ModelOption | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if not self.env.logging_file.stem.endswith(self.tag):
            self.env.logging_file = self.env.logging_file.with_stem(f"{self.env.logging_file.stem}-{self.tag}")
        if not self.env.argument_file.stem.endswith(self.tag):
            self.env.argument_file = self.env.argument_file.with_stem(f"{self.env.argument_file.stem}-{self.tag}")

        self.env.output_home = self.env.output_home or Path("output")
        if self.data and self.model:
            self.env.output_home = self.model.finetuning_home / self.data.name
        elif self.data:
            self.env.output_home = self.env.output_home / self.data.home
        configure_dual_logger(level=self.env.msg_level, fmt=self.env.msg_format, datefmt=self.env.date_format,
                              filename=self.env.output_home / self.env.logging_file)

    def reconfigure_output(self, version=None):
        if not version:
            version = now('%m%d.%H%M%S')
        self.env.csv_logger = CSVLogger(self.model.finetuning_home, name=self.data.name,
                                        version=f'{self.tag}-{self.env.job_name}-{version}',
                                        flush_logs_every_n_steps=1)

        existing_file = self.env.output_home / self.env.logging_file
        existing_content = existing_file.read_text() if existing_file.exists() else None
        self.env.output_home = Path(self.env.csv_logger.log_dir)
        configure_dual_logger(level=self.env.msg_level, fmt=self.env.msg_format, datefmt=self.env.date_format,
                              filename=self.env.output_home / self.env.logging_file, existing_content=existing_content)
        existing_file.unlink(missing_ok=True)
        return self

    def save_arguments(self, to: Path | str = None) -> Path | None:
        if not self.env.output_home:
            return None
        args_file = to if to else self.env.output_home / self.env.argument_file
        args_json = self.to_json(default=str, ensure_ascii=False, indent=2)
        make_parent_dir(args_file).write_text(args_json, encoding="utf-8")
        return args_file

    def info_arguments(self):
        table = str_table(self.dataframe(), tablefmt="presto")  # "plain", "presto"
        for line in table.splitlines() + [hr(c='-')]:
            logger.info(line)
        return self

    def dataframe(self, columns=None):
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
            to_dataframe(columns=columns, raw=self.prog, data_prefix="prog"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.model, data_prefix="model"),
        ]).reset_index(drop=True)

    def show(self):
        out_hr(c='-')
        out_table(self.dataframe())
        out_hr(c='-')
        return self


@dataclass
class ServerArguments(CommonArguments):
    tag = "serve"

    def __post_init__(self):
        super().__post_init__()
        if self.tag in ("serve", "test"):
            assert self.model.finetuning_home.exists() and self.model.finetuning_home.is_dir(), \
                f"No finetuning home: {self.model.finetuning_home}"
            if not self.model.finetuning_name:
                ckpt_files: List[Path] = files(self.env.output_home / "**/*.ckpt")
                assert ckpt_files, f"No checkpoint file in {self.model.finetuning_home}"
                ckpt_files = sorted([x for x in ckpt_files if "temp" not in str(x) and "tmp" not in str(x)], key=str)
                self.model.finetuning_name = ckpt_files[-1].relative_to(self.env.output_home)
            assert (self.env.output_home / self.model.finetuning_name).exists() and (self.env.output_home / self.model.finetuning_name).is_file(), \
                f"No checkpoint file: {self.env.output_home / self.model.finetuning_name}"


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
    def __init__(self, args: CommonArguments, delete_on_exit: bool = True):
        self.args: CommonArguments = args
        self.delete_on_exit: bool = delete_on_exit

    def __enter__(self) -> Path:
        self.args_file: Path | None = self.args.save_arguments()
        return self.args_file

    def __exit__(self, *exc_info):
        if self.delete_on_exit and self.args_file:
            self.args_file.unlink(missing_ok=True)


class RuntimeChecking:
    def __init__(self, args: CommonArguments):
        self.args: CommonArguments = args

    def __enter__(self):
        self.args.time.set_started()

    def __exit__(self, *exc_info):
        self.args.time.set_settled()
        self.args.save_arguments(self.args.env.output_home / self.args.env.argument_file)
