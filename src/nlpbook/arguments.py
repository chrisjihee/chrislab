import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import pytorch_lightning
import transformers
from dataclasses_json import DataClassJsonMixin
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from transformers import PretrainedConfig

from chrisbase.data import ProjectEnv, OptionData, ResultData, CommonArguments
from chrisbase.io import files, LoggingFormat
from chrisbase.time import now
from chrisbase.util import to_dataframe

logger = logging.getLogger(__name__)


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
    num_check: int = field(default=3)

    # for research
    # num_entity: int | None = field(default=None)
    # num_relation: int | None = field(default=None)

    def __post_init__(self):
        if self.home:
            self.home = Path(self.home).absolute()


@dataclass
class ModelOption(OptionData):
    pretrained: str | Path = field()
    finetuning: str | Path = field()
    seq_len: int = field(default=128)  # maximum total input sequence length after tokenization
    config: PretrainedConfig | None = field(default=None)
    name: str | Path | None = field(default=None)

    # for research
    # src_max_length: int = field(default=512)
    # train_tgt_max_length: int = field(default=512)
    # eval_tgt_max_length: int = field(default=90)
    # src_descrip_max_length: int = field(default=0)
    # tgt_descrip_max_length: int = field(default=0)
    # seq_dropout: float = field(default=0.1)
    # decoder: str = field(default="beam_search")  # "[beam_search, diverse_beam_search, do_sample]"
    # num_beams: int = field(default=40)
    # num_beam_groups: int = field(default=1)
    # diversity_penalty: float = field(default=0.0)
    # use_prefix_search: bool = field(default=False)

    def __post_init__(self):
        self.finetuning = Path(self.finetuning).absolute()


@dataclass
class HardwareOption(OptionData):
    cpu_workers: int = field(default=os.cpu_count())
    train_batch: int = field(default=32)
    infer_batch: int = field(default=32)
    accelerator: str = field(default="auto")  # possbile value: "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"
    precision: int | str = field(default=32)  # floating-point precision type
    strategy: str = field(default="auto")  # multi-device strategies
    devices: List[int] | int | str = field(default="auto")  # devices to use

    def __post_init__(self):
        if not self.strategy:
            if self.devices == 1 or isinstance(self.devices, list) and len(self.devices) == 1:
                self.strategy = "single_device"
            elif isinstance(self.devices, int) and self.devices > 1 or isinstance(self.devices, list) and len(self.devices) > 1:
                self.strategy = "ddp"


@dataclass
class LearningOption(OptionData):
    random_seed: int | None = field(default=None)
    optimizer_cls: str = field(default="Adam")
    learning_rate: float = field(default=5e-5)
    saving_mode: str = field(default="min val_loss")
    num_saving: int = field(default=3)
    num_epochs: int = field(default=1)
    log_text: bool = field(default=False)
    check_rate_on_training: float = field(default=1.0)
    print_rate_on_training: float = field(default=0.0333)
    print_rate_on_validate: float = field(default=0.334)
    print_rate_on_evaluate: float = field(default=0.334)
    print_step_on_training: int = field(default=-1)
    print_step_on_validate: int = field(default=-1)
    print_step_on_evaluate: int = field(default=-1)
    tag_format_on_training: str = field(default="")
    tag_format_on_validate: str = field(default="")
    tag_format_on_evaluate: str = field(default="")
    name_format_on_saving: str = field(default="")

    def __post_init__(self):
        self.check_rate_on_training = abs(self.check_rate_on_training)
        self.print_rate_on_training = abs(self.print_rate_on_training)
        self.print_rate_on_validate = abs(self.print_rate_on_validate)
        self.print_rate_on_evaluate = abs(self.print_rate_on_evaluate)


@dataclass
class ProgressChecker(ResultData):
    result: dict = field(init=False, default_factory=dict)
    tb_logger: TensorBoardLogger = field(init=False, default=None)
    csv_logger: CSVLogger = field(init=False, default=None)
    world_size: int = field(init=False, default=1)
    node_rank: int = field(init=False, default=0)
    local_rank: int = field(init=False, default=0)
    global_rank: int = field(init=False, default=0)
    global_step: int = field(init=False, default=0)
    global_epoch: float = field(init=False, default=0.0)


@dataclass
class MLArguments(CommonArguments):
    tag = None
    prog: ProgressChecker = field(default_factory=ProgressChecker)
    data: DataOption | None = field(default=None)
    model: ModelOption | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def configure_csv_logger(self, version=None):  # TODO: Remove someday
        if not version:
            version = now('%m%d.%H%M%S')
        self.prog.csv_logger = CSVLogger(self.model.finetuning, name=self.data.name,
                                         version=f'{self.tag}-{self.env.job_name}-{version}',
                                         flush_logs_every_n_steps=1)
        self.env.output_home = Path(self.prog.csv_logger.log_dir)
        return self

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.prog, data_prefix="prog"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data") if self.data else None,
            to_dataframe(columns=columns, raw=self.model, data_prefix="model") if self.model else None,
        ]).reset_index(drop=True)
        return df


@dataclass
class ServerArguments(MLArguments):
    tag = "serve"

    def __post_init__(self):
        super().__post_init__()
        if self.tag in ("serve", "test"):
            assert self.model.finetuning.exists() and self.model.finetuning.is_dir(), \
                f"No finetuning home: {self.model.finetuning}"
            if not self.model.name:
                ckpt_files: List[Path] = files(self.env.output_home / "**/*.ckpt")
                assert ckpt_files, f"No checkpoint file in {self.env.output_home}"
                ckpt_files = sorted([x for x in ckpt_files if "temp" not in str(x) and "tmp" not in str(x)], key=str)
                self.model.name = ckpt_files[-1].relative_to(self.env.output_home)
            elif (self.env.output_home / self.model.name).exists() and (self.env.output_home / self.model.name).is_dir():
                ckpt_files: List[Path] = files(self.env.output_home / self.model.name / "**/*.ckpt")
                assert ckpt_files, f"No checkpoint file in {self.env.output_home / self.model.name}"
                ckpt_files = sorted([x for x in ckpt_files if "temp" not in str(x) and "tmp" not in str(x)], key=str)
                self.model.name = ckpt_files[-1].relative_to(self.env.output_home)
            assert (self.env.output_home / self.model.name).exists() and (self.env.output_home / self.model.name).is_file(), \
                f"No checkpoint file: {self.env.output_home / self.model.name}"

    @staticmethod
    def from_args(  # TODO: Remove someday
            # env
            project: str = None,
            job_name: str = None,
            debugging: bool = False,
            # data
            data_home: str = "data",
            data_name: str = None,
            train_file: str = None,
            valid_file: str = None,
            test_file: str = None,
            num_check: int = 2,
            # model
            pretrained: str = "klue/roberta-small",
            finetuning: str = "finetuning",
            model_name: str = None,
            seq_len: int = 128,
            # hardware
            accelerator: str = "gpu",
            precision: str = "32-true",
            strategy: str = "auto",
            device: List[int] = (0,),
            batch_size: int = 100,
    ) -> "ServerArguments":
        pretrained = Path(pretrained)
        return ServerArguments(
            env=ProjectEnv(
                project=project,
                job_name=job_name if job_name else pretrained.name,
                debugging=debugging,
                msg_level=logging.DEBUG if debugging else logging.INFO,
                msg_format=LoggingFormat.DEBUG_36 if debugging else LoggingFormat.CHECK_40,
            ),
            data=DataOption(
                home=data_home,
                name=data_name,
                files=DataFiles(
                    train=train_file,
                    valid=valid_file,
                    test=test_file,
                ),
                num_check=num_check,
            ),
            model=ModelOption(
                pretrained=pretrained,
                finetuning=finetuning,
                name=model_name,
                seq_len=seq_len,
            ),
        )


@dataclass
class TesterArguments(ServerArguments):
    tag = "test"
    hardware: HardwareOption = field(default_factory=HardwareOption, metadata={"help": "device information"})

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.hardware, data_prefix="hardware"),
        ]).reset_index(drop=True)
        return df

    @staticmethod
    def from_args(  # TODO: Remove someday
            # env
            project: str = None,
            job_name: str = None,
            debugging: bool = False,
            # data
            data_home: str = "data",
            data_name: str = None,
            train_file: str = None,
            valid_file: str = None,
            test_file: str = None,
            num_check: int = 2,
            # model
            pretrained: str = "klue/roberta-small",
            finetuning: str = "finetuning",
            model_name: str = None,
            seq_len: int = 128,
            # hardware
            train_batch: int = 100,
            infer_batch: int = 100,
            accelerator: str = "gpu",
            precision: str = "32-true",
            strategy: str = "auto",
            device: List[int] = (0,),
    ) -> "TesterArguments":
        pretrained = Path(pretrained)
        return TesterArguments(
            env=ProjectEnv(
                project=project,
                job_name=job_name if job_name else pretrained.name,
                debugging=debugging,
                msg_level=logging.DEBUG if debugging else logging.INFO,
                msg_format=LoggingFormat.DEBUG_36 if debugging else LoggingFormat.CHECK_40,
            ),
            data=DataOption(
                home=data_home,
                name=data_name,
                files=DataFiles(
                    train=train_file,
                    valid=valid_file,
                    test=test_file,
                ),
                num_check=num_check,
            ),
            model=ModelOption(
                pretrained=pretrained,
                finetuning=finetuning,
                name=model_name,
                seq_len=seq_len,
            ),
            hardware=HardwareOption(
                train_batch=train_batch,
                infer_batch=infer_batch,
                accelerator=accelerator,
                precision=precision,
                strategy=strategy,
                devices=device,
            ),
        )


@dataclass
class TrainerArguments(TesterArguments):
    tag = "train"
    learning: LearningOption = field(default_factory=LearningOption)

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        df = pd.concat([
            super().dataframe(columns=columns),
            to_dataframe(columns=columns, raw=self.learning, data_prefix="learning"),
        ]).reset_index(drop=True)
        return df

    def set_seed(self) -> None:  # TODO: Remove someday
        if self.learning.random_seed is not None:
            transformers.set_seed(self.learning.random_seed)
            pytorch_lightning.seed_everything(self.learning.random_seed)
        else:
            logger.warning("not fixed seed")

    @staticmethod
    def from_args(  # TODO: Remove someday
            # env
            project: str = None,
            job_name: str = None,
            debugging: bool = False,
            # data
            data_home: str = "data",
            data_name: str = None,
            train_file: str = None,
            valid_file: str = None,
            test_file: str = None,
            num_check: int = 2,
            # model
            pretrained: str = "klue/roberta-small",
            finetuning: str = "finetuning",
            model_name: str = None,
            seq_len: int = 128,
            src_max_length: int = 512,
            train_tgt_max_length: int = 512,
            eval_tgt_max_length: int = 90,
            src_descrip_max_length: int = 0,
            tgt_descrip_max_length: int = 0,
            seq_dropout: float = 0.1,
            decoder: str = "beam_search",
            num_beams: int = 40,
            num_beam_groups: int = 1,
            diversity_penalty: float = 0.0,
            use_prefix_search: bool = False,
            # hardware
            train_batch: int = 100,
            infer_batch: int = 100,
            accelerator: str = "gpu",
            precision: str = "32-true",
            strategy: str = "auto",
            device: List[int] = (0,),
            # learning
            optimizer_cls: str = "Adam",
            learning_rate: float = 5e-5,
            saving_policy: str = None,
            num_saving: int = 1,
            num_epochs: int = 1,
            log_text: bool = False,
            check_rate_on_training: float = 0.2,
            print_rate_on_training: float = 0.0333,
            print_rate_on_validate: float = 0.334,
            print_rate_on_evaluate: float = 0.334,
            print_step_on_training: int = -1,
            print_step_on_validate: int = -1,
            print_step_on_evaluate: int = -1,
            tag_format_on_training: str = "",
            tag_format_on_validate: str = "",
            tag_format_on_evaluate: str = "",
            name_format_on_saving: str = "",
            seed: int = 7,
    ) -> "TrainerArguments":
        pretrained = Path(pretrained)
        return TrainerArguments(
            env=ProjectEnv(
                project=project,
                job_name=job_name if job_name else pretrained.name,
                debugging=debugging,
                msg_level=logging.DEBUG if debugging else logging.INFO,
                msg_format=LoggingFormat.DEBUG_36 if debugging else LoggingFormat.CHECK_40,
            ),
            data=DataOption(
                home=data_home,
                name=data_name,
                files=DataFiles(
                    train=train_file,
                    valid=valid_file,
                    test=test_file,
                ),
                num_check=num_check,
            ),
            model=ModelOption(
                pretrained=pretrained,
                finetuning=finetuning,
                name=model_name,
                seq_len=seq_len,
                # src_max_length=src_max_length,
                # train_tgt_max_length=train_tgt_max_length,
                # eval_tgt_max_length=eval_tgt_max_length,
                # src_descrip_max_length=src_descrip_max_length,
                # tgt_descrip_max_length=tgt_descrip_max_length,
                # seq_dropout=seq_dropout,
                # decoder=decoder,
                # num_beams=num_beams,
                # num_beam_groups=num_beam_groups,
                # diversity_penalty=diversity_penalty,
                # use_prefix_search=use_prefix_search,
            ),
            hardware=HardwareOption(
                train_batch=train_batch,
                infer_batch=infer_batch,
                accelerator=accelerator,
                precision=precision,
                strategy=strategy,
                devices=device,
            ),
            learning=LearningOption(
                random_seed=seed,
                optimizer_cls=optimizer_cls,
                learning_rate=learning_rate,
                saving_mode=saving_policy,
                num_saving=num_saving,
                num_epochs=num_epochs,
                log_text=log_text,
                check_rate_on_training=check_rate_on_training,
                print_rate_on_training=print_rate_on_training,
                print_rate_on_validate=print_rate_on_validate,
                print_rate_on_evaluate=print_rate_on_evaluate,
                print_step_on_training=print_step_on_training,
                print_step_on_validate=print_step_on_validate,
                print_step_on_evaluate=print_step_on_evaluate,
                tag_format_on_training=tag_format_on_training,
                tag_format_on_validate=tag_format_on_validate,
                tag_format_on_evaluate=tag_format_on_evaluate,
                name_format_on_saving=name_format_on_saving,
            ),
        )
