import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dataclasses_json import DataClassJsonMixin

from chrisbase.io import ProjectEnv
from chrisbase.io import files, make_parent_dir, out_hr, out_table
from chrisbase.util import to_dataframe


@dataclass
class ClassificationArguments(DataClassJsonMixin):
    def env_data(self) -> ProjectEnv:
        if isinstance(self.env, ProjectEnv):
            return self.env
        else:
            return ProjectEnv.from_dict(self.env)

    def env_dict(self) -> dict:
        if isinstance(self.env, DataClassJsonMixin):
            return self.env.to_dict()
        else:
            return self.env

    def __post_init__(self):
        self.env = self.env_data()
        self.working_config_file = self.env.running_file.with_suffix('.json').name
        self.downstream_model_home = Path(self.downstream_model_home)

    env: ProjectEnv | dict = field(
        metadata={"help": "current project environment"}
    )
    pretrained_model_path: Path | str | None = field(
        default="beomi/kcbert-base",
        metadata={"help": "name/path of pretrained model"}
    )
    downstream_model_home: Path | str | None = field(
        default=None,
        metadata={"help": "root directory of output model and working config"}
    )
    downstream_model_file: str | None = field(
        default=None,
        metadata={"help": "filename or filename format of output model"}
    )
    downstream_task_name: str = field(
        default="document-classification",
        metadata={"help": "name of downstream task"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization. "
                          "Sequences longer than this will be truncated, sequences shorter will be padded."}
    )
    working_config_file: str | None = field(
        init=False,
        metadata={"help": "filename of current config"}
    )

    def save_working_config(self, to: Path | str = None) -> Path:
        self.env = self.env_dict()
        config_file = make_parent_dir(to) if to \
            else make_parent_dir(self.downstream_model_home / self.working_config_file)
        config_file.write_text(self.to_json(default=str, ensure_ascii=False, indent=2))
        return config_file

    def as_dataframe(self):
        columns = [self.__class__.__name__, "value"]
        return pd.concat([
            to_dataframe(data_prefix="env", raw=self.env, columns=columns),
            to_dataframe(data_exclude="env", raw=self, columns=columns),
        ]).reset_index(drop=True)

    def print_dataframe(self):
        out_hr(c='-')
        out_table(self.as_dataframe())
        out_hr(c='-')
        return self


@dataclass
class ClassificationTrainArguments(ClassificationArguments):
    def __post_init__(self):
        super().__post_init__()
        if not self.save_top_k:
            self.save_top_k = self.epochs
        self.downstream_data_home = Path(self.downstream_data_home)

    downstream_data_home: Path | str | None = field(
        default="/content/Korpora",
        metadata={"help": "root of downstream data"}
    )
    downstream_data_name: str | None = field(
        default=None,
        metadata={"help": "name of downstream data"}
    )
    save_top_k: int = field(
        default=None,
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
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "overwrite the cached training and evaluation sets"}
    )
    force_download: bool = field(
        default=False,
        metadata={"help": "force to download downstream data and pretrained models"}
    )
    test_mode: bool = field(
        default=False,
        metadata={"help": "test mode enables `fast_dev_run`"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "learning rate"}
    )
    epochs: int = field(
        default=1,
        metadata={"help": "max epochs"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "batch size. if 0, let lightening find the best batch size"}
    )
    cpu_workers: int = field(
        default=os.cpu_count(),
        metadata={"help": "number of CPU workers"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "enable train on floating point 16"}
    )


@dataclass
class ClassificationDeployArguments(ClassificationArguments):
    def __post_init__(self):
        super().__post_init__()
        if not self.downstream_model_file:
            assert self.downstream_model_home.exists() and self.downstream_model_home.is_dir(), \
                f"downstream_model_path is not a directory: {self.downstream_model_home}"
            ckpt_files = files(self.downstream_model_home / "*.ckpt")
            ckpt_files = sorted([x for x in ckpt_files if "temp" not in str(x) and "tmp" not in str(x)], key=str)
            assert len(ckpt_files) > 0, f"No checkpoint file in {self.downstream_model_home}"
            self.downstream_model_file = ckpt_files[-1].name
