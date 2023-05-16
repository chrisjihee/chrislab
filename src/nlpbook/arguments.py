from datetime import datetime, timedelta
import os
from dataclasses import dataclass, field
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
    train: str | Path | None = field(default=None, metadata={"help": "data file for 'train' split"})
    valid: str | Path | None = field(default=None, metadata={"help": "data file for 'valid' split"})
    test: str | Path | None = field(default=None, metadata={"help": "data file for 'test' split"})


@dataclass
class NLUArguments(DataClassJsonMixin):
    t1 = datetime.now()
    t2 = datetime.now()
    action = "base"
    env: ProjectEnv = field(
        metadata={"help": "current project environment"}
    )
    downstream_data_file: DataFiles | None = field(
        default=None,
        metadata={"help": "filenames of downstream data"}
    )
    downstream_data_name: str | Path | None = field(
        default=None,
        metadata={"help": "name of downstream data"}
    )
    downstream_data_home: str | Path | None = field(
        default=None,
        metadata={"help": "root of downstream data"}
    )
    downstream_data_caching: bool = field(
        default=False,
        metadata={"help": "overwrite the cached training and evaluation sets"}
    )
    downstream_data_download: bool = field(
        default=False,
        metadata={"help": "force to download downstream data and pretrained models"}
    )
    downstream_model_file: str | Path | None = field(
        default=None,
        metadata={"help": "filename or filename format of downstream model"}
    )
    downstream_model_home: str | Path | None = field(
        default=None,
        metadata={"help": "root directory of output model and working config"}
    )
    pretrained_model_path: str | Path = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "name/path of pretrained model"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    working_config_file: str | Path | None = field(
        default="config.json",
        metadata={"help": "filename of current config"}
    )
    started_time: str | None = field(init=False)
    settled_time: str | None = field(init=False)
    elapsed_time: str | None = field(init=False)

    def __post_init__(self):
        assert isinstance(self.downstream_model_home, (str, Path)), "downstream_model_home must be str or Path"
        self.downstream_data_home = Path(self.downstream_data_home)
        self.downstream_model_home = make_dir(self.downstream_model_home)
        self.working_config_file = Path(self.working_config_file)
        if not self.working_config_file.stem.endswith(self.action):
            self.working_config_file = self.working_config_file.with_stem(f"{self.working_config_file.stem}-{self.action}")
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

    def downstream_data_file_as_obj(self) -> DataFiles:
        if isinstance(self.downstream_data_file, dict):
            return DataFiles.from_dict(self.downstream_data_file)
        if isinstance(self.downstream_data_file, DataFiles):
            return self.downstream_data_file

    def save_working_config(self, to: Path | str = None) -> Path:
        config_file = to if to else self.downstream_model_home / self.downstream_data_name / self.working_config_file
        make_parent_dir(config_file).write_text(self.to_json(default=str, ensure_ascii=False, indent=2))
        return config_file

    def as_dataframe(self):
        columns = [self.__class__.__name__, "value"]
        return pd.concat([
            to_dataframe(data_prefix="env", raw=self.env, columns=columns),
            to_dataframe(data_prefix="downstream_data_file", raw=self.downstream_data_file, columns=columns),
            to_dataframe(data_exclude=("env", "downstream_data_file"), raw=self, columns=columns),
        ]).reset_index(drop=True)

    def print_dataframe(self):
        out_hr(c='-')
        out_table(self.as_dataframe())
        out_hr(c='-')
        return self


@dataclass
class NLUServerArguments(NLUArguments):
    action = "serve"

    def __post_init__(self):
        super().__post_init__()
        if self.action in ("serve", "test"):
            assert self.downstream_model_home.exists() and self.downstream_model_home.is_dir(), \
                f"No model directory: {self.downstream_model_home}"
            if not self.downstream_model_file:
                ckpt_files = files(self.downstream_model_home / "*.ckpt")
                ckpt_files = sorted([x for x in ckpt_files if "temp" not in str(x) and "tmp" not in str(x)], key=str)
                assert len(ckpt_files) > 0, f"No checkpoint file in {self.downstream_model_home}"
                self.downstream_model_file = ckpt_files[-1].name
            assert (self.downstream_model_home / self.downstream_model_file).exists(), \
                f"No checkpoint file: {self.downstream_model_home / self.downstream_model_file}"


@dataclass
class NLUTesterArguments(NLUServerArguments):
    action = "test"
    batch_size: int = field(
        default=32,
        metadata={"help": "batch size. if 0, let lightening find the best batch size"}
    )
    cpu_workers: int = field(
        default=os.cpu_count(),
        metadata={"help": "number of CPU workers"}
    )
    accelerator: str | Accelerator = field(
        default="auto",
        metadata={"help": 'accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")'}
    )
    precision: int | str = field(
        default=32,
        metadata={"help": "floating-point precision type"}
    )
    strategy: str | Strategy | None = field(
        default=None,
        metadata={"help": 'training strategies'}
    )
    devices: List[int] | int | str = field(
        default=1,
        metadata={"help": 'devices to use'}
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.strategy:
            if self.devices == 1 or isinstance(self.devices, list) and len(self.devices) == 1:
                self.strategy = "single_device"
            elif isinstance(self.devices, int) and self.devices > 1 or isinstance(self.devices, list) and len(self.devices) > 1:
                self.strategy = "ddp"


@dataclass
class NLUTrainerArguments(NLUTesterArguments):
    action = "train"
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "learning rate"}
    )
    epochs: int = field(
        default=1,
        metadata={"help": "max epochs"}
    )
    log_steps: int = field(
        default=10,
        metadata={"help": "check interval"}
    )
    save_top_k: int = field(
        default=100,
        metadata={"help": "save top k model checkpoints"}
    )
    monitor: str = field(
        default="min val_loss",
        metadata={"help": "monitor condition (save top k)"}
    )
    seed: int | None = field(
        default=None,
        metadata={"help": "random seed"}
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.save_top_k:
            self.save_top_k = self.epochs


class CommandConfig:
    def __init__(self, args: NLUArguments):
        self.args: NLUArguments = args

    def __enter__(self) -> Path:
        self.config: Path = self.args.save_working_config()
        return self.config

    def __exit__(self, *exc_info):
        self.config.unlink(missing_ok=True)


class RuntimeCheckingOnArgs:
    def __init__(self, args: NLUArguments, log_dir: str):
        self.args: NLUArguments = args
        self.log_dir: Path = Path(log_dir)

    def __enter__(self):
        self.args.set_started_time()

    def __exit__(self, *exc_info):
        self.args.set_settled_time()
        self.args.save_working_config(self.log_dir / self.args.working_config_file)
