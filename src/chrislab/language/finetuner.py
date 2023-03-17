"""
### 참고 자료

##### 1. 논문
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

##### 2. 코드
   - [Text classification examples (transformers)](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)
   - [Image classification examples (lightning)](https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/image_classifier)
   - [Finetuning (KoELECTRA)](https://github.com/monologg/KoELECTRA/tree/master/finetune)
   - [KE-T5 Downstreams](https://github.com/AIRC-KETI/ke-t5-downstreams)
   - [Process (datasets)](https://huggingface.co/docs/datasets/process)
"""
from __future__ import annotations

from collections import Counter
from itertools import chain
from pathlib import Path
from random import Random
from sys import stderr, stdout
from typing import Dict, Optional

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset, DatasetDict, load_dataset, DownloadMode
from datasets.formatting.formatting import LazyBatch
from datasets.metric import Metric
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.strategies import DataParallelStrategy, DDPStrategy, DeepSpeedStrategy
from tokenizers import ByteLevelBPETokenizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import TextInput, BatchEncoding, TruncationStrategy
from transformers.utils import PaddingStrategy

from chrisbase.io import JobTimer, load_attrs, merge_attrs, merge_dicts, copy_dict, set_tokenizers_parallelism, include_cuda_bin_dir, set_torch_ext_path, file_table, make_dir, new_path, save_attrs, remove_dir_check
from chrisbase.util import tupled, append_intersection, no_space, no_replacement, no_nonprintable, display_histogram, to_morphemes, OK
from chrisdict import AttrDict
from .modeling import BertHeadModel, T5HeadModel, additive_tokens_for_morp_tag
from ..common.tokenizer_korbert import KorbertTokenizer
from ..common.util import StageMarker, MuteDatasetProgress, time_tqdm_cls, mute_tqdm_cls, to_tensor_batch, limit_num_samples

finetuning_class_mapping = {
    'BertHeadModel': BertHeadModel,
    'T5HeadModel': T5HeadModel,
}


class MyFinetuner(Fabric):
    """
    Finetuner for sentence-level classification or regression tasks.
    - Refer to `lightning.fabric.Fabric`
    """

    def __init__(
            self, config, prefix=None, postfix=None, save_cache=True, reset_cache=False,
            db_host="localhost", db_port=6382, milestones=("INIT", "TRAIN", "METER", "SAVE"),
    ):
        self.state: AttrDict = load_attrs(config)
        self.state = merge_attrs(self.state, post={
            "devices": tupled(self.state.devices) if self.state.devices else None,
            "pretrained": merge_dicts(copy_dict(self.state.pretrained), {"path": Path(self.state.pretrained.path)}),
            "label_column": self.state.label_column if "label_column" in self.state else "label",
            "cached_home": Path(self.state.cached_home) if self.state.cached_home else None,
            "dataset_home": Path(self.state.dataset_home) if self.state.dataset_home else None,
            "finetuned_home": Path(self.state.finetuned_home) if self.state.finetuned_home else None,
            "predicted_home": Path(self.state.predicted_home) if self.state.predicted_home else None,
        })

        seed_everything(self.state.seed)
        self.rand: Random = Random(self.state.seed)
        self.prefix: str | None = prefix if prefix and len(prefix) > 0 else None
        self.postfix: str | None = postfix if postfix and len(postfix) > 0 else None
        self.save_cache: bool = save_cache
        self.reset_cache: bool = reset_cache
        self.finetuning_model: BertHeadModel | T5HeadModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.tok_coder: ByteLevelBPETokenizer | None = None
        self.optimizer: Optimizer | None = None
        self.scheduler: object | None = None
        self.dataloader: Dict[str, DataLoader] = {}
        self.loss_metric: nn.modules.loss._Loss | None = None
        self.score_metric: Metric | None = None
        self.input_datasets: DatasetDict | None = None
        self.sample_dataset: Dataset | None = None
        self.time_tqdm = time_tqdm_cls(bar_size=30, desc_size=17, prefix=self.prefix, file=stdout)
        self.mute_tqdm = mute_tqdm_cls()
        self.db_host = db_host
        self.db_port = db_port
        self.milestones = milestones
        self.cache_dirs = [self.state.cached_home]
        set_tokenizers_parallelism(False)
        if self.state.devices:
            include_cuda_bin_dir()
            set_torch_ext_path(dev=self.state.devices[0])
        super(MyFinetuner, self).__init__(precision=self.state.precision if 'precision' in self.state else 32,
                                          devices=self.state.devices if 'devices' in self.state else None,
                                          accelerator='auto', strategy=self.configure_strategy())
        self.state.device = self.device

    def show_state_values(self, verbose=True):
        with JobTimer(verbose=verbose, rb=1) as timer:
            file_table(pd.DataFrame(
                chain.from_iterable([
                    [(f'{k}[{s}]', f'({type(v).__qualname__}) {v}' if v is not None else 'None') for s, v in self.state[k].items()] if k == 'data_files' else
                    [(k, f'({type(self.state[k]).__qualname__}) {self.state[k]}' if self.state[k] is not None else 'None')]
                    for k in self.state.log_targets if k in self.state]),
                columns=["key", "value"]), showindex=False, file=timer.file)

    def ready(self, show_state=False, draw_figure=False) -> None:
        with JobTimer(f"Preparing({self.state.data_name}/{self.state.data_part})", prefix=self.prefix, postfix=self.postfix, mb=1, rt=1, rb=1, rc='=', verbose=self.is_global_zero):
            # BEGIN
            print(f"cache cleared : {OK(all(remove_dir_check(x, real=self.reset_cache) for x in self.cache_dirs))}")
            if show_state:
                with JobTimer(verbose=self.is_global_zero):
                    self.show_state_values(verbose=self.is_global_zero)
                    assert self.state.data_name and isinstance(self.state.data_name, (Path, str)), f"Invalid data_name: ({type(self.state.data_name).__qualname__}) {self.state.data_name}"

            # READY(data)
            with JobTimer(verbose=self.is_global_zero):
                self.prepare_datasets(verbose=self.is_global_zero, draw_figure=draw_figure)
                self.check_tokenizer(sample=self.is_global_zero)

    def run(self, show_state=True) -> None:
        with JobTimer(f"Finetuning({self.state.data_name}/{self.state.data_part})", prefix=self.prefix, postfix=self.postfix, mb=1, rt=1, rb=1, rc='=', verbose=self.is_global_zero, file=stdout, flush_sec=0.3):
            with JobTimer(f"Finetuning({self.state.data_name}/{self.state.data_part})", prefix=self.prefix, postfix=self.postfix, mb=1, rt=1, rb=1, rc='=', verbose=self.is_global_zero, file=stderr, flush_sec=0.3, pb=1):
                # BEGIN
                print(f"cache cleared : {OK(all(remove_dir_check(x, real=self.reset_cache) for x in self.cache_dirs))}")
                if show_state:
                    with JobTimer(verbose=self.is_global_zero):
                        self.show_state_values(verbose=self.is_global_zero)
                        assert self.state.data_name and isinstance(self.state.data_name, (Path, str)), f"Invalid data_name: ({type(self.state.data_name).__qualname__}) {self.state.data_name}"

                # READY(data)
                with JobTimer(verbose=self.is_global_zero):
                    self.prepare_datasets(verbose=self.is_global_zero, draw_figure=False)
                    self.prepare_dataloader()

                # READY(finetuning)
                assert 'train' in self.dataloader, f"No train dataloader in {list(self.dataloader.keys())}"
                self.state.steps_per_epoch = len(self.dataloader['train'])
                self.state.total_steps = self.state.num_train_epochs * self.state.steps_per_epoch
                epoch_per_step = 1.0 / self.state.steps_per_epoch
                self.finetuning_model = finetuning_class_mapping[self.state.finetuning_class](state=self.state, tokenizer=self.tokenizer)
                with JobTimer(verbose=self.is_global_zero, rb=1):
                    self.check_pretrained(sample=self.is_global_zero)

                with JobTimer(verbose=self.is_global_zero, rb=1 if self.state.strategy == 'deepspeed' else 0):
                    self.optimizer = self.configure_optimizer()
                    self.scheduler = self.configure_scheduler()
                    self.loss_metric = self.configure_loss()
                    self.score_metric = evaluate.load(self.state.score_metric.major, self.state.score_metric.minor)
                    self.state['finetuning_model'] = f"{type(self.finetuning_model).__qualname__} | pretrained={self.state.pretrained.path.name}"
                    self.state['optimizer'] = f"{type(self.optimizer).__qualname__} | lr={self.state.learning_rate}"
                    self.state['scheduler'] = f"{type(self.scheduler).__qualname__} | gamma={self.state.lr_scheduler_gamma}"
                    self.state['loss_metric'] = f"{type(self.loss_metric).__qualname__}"
                    self.state['score_metric'] = f"{self.state.score_metric.major}/{self.state.score_metric.minor}"
                    for k in ('finetuning_model', 'optimizer', 'scheduler', 'loss_metric', 'score_metric'):
                        print(f"- {k:30s} = {self.state[k]}")
                with JobTimer(verbose=self.is_global_zero, rb=1, rc='='):
                    self.finetuning_model, self.optimizer = self.setup(self.finetuning_model, self.optimizer)

                # READY(output)
                assert self.state.finetuned_home and isinstance(self.state.finetuned_home, Path), f"Invalid finetuned_home: ({type(self.state.finetuned_home).__qualname__}) {self.state.finetuned_home}"
                assert isinstance(self.state.finetuned_sub, (type(None), Path, str)), f"Invalid finetuned_sub: ({type(self.state.finetuned_sub).__qualname__}) {self.state.finetuned_sub}"
                tab_name: str = self.state.finetuned_sub if self.state.finetuned_sub else "default"
                with StageMarker(self.global_rank, self.world_size, self.milestones,
                                 db_name=self.state.data_name, tab_name=tab_name, host=self.db_host, port=self.db_port) as marker:
                    marker.clear()
                finetuned_dir: Path = make_dir(self.state.finetuned_home / self.state.data_name / tab_name)
                finetuned_files = {
                    "state": finetuned_dir / "finetuner_state.json",
                    "model": finetuned_dir / "pytorch_model",
                }
                logs = {
                    "step": 0,
                    "epoch": 0.0,
                    "record": []
                }
                self.state.records = logs["record"]

                # EPOCH
                with StageMarker(self.global_rank, self.world_size, self.milestones,
                                 db_name=self.state.data_name, tab_name=tab_name, host=self.db_host, port=self.db_port) as marker:
                    for epoch in range(1, self.state.num_train_epochs + 1):
                        with JobTimer(verbose=True, rb=1 if self.is_global_zero and epoch < self.state.num_train_epochs else 0):
                            # INIT
                            current = f"(Epoch {epoch:02d})"
                            marker.initialize(stage=current)
                            metrics = {}
                            with JobTimer(verbose=self.is_global_zero):
                                print(self.time_tqdm.to_desc(pre=current, desc=f"composed #{self.global_rank + 1:01d}") + f": learning_rate={self.get_learning_rate():.10f}")
                            marker.mark_done("INIT", stage=current, state_table_file=stderr)

                            # TRAIN
                            self.finetuning_model.train()
                            with torch.enable_grad():
                                for k in self.input_datasets.keys():
                                    if k not in self.dataloader or not self.dataloader[k]:
                                        continue
                                    if k not in self.state.finetuning_splits or not self.state.finetuning_splits[k]:
                                        continue
                                    outputs = []
                                    dataloader = self.dataloader['train']
                                    with JobTimer() as timer:
                                        tqdm = self.time_tqdm if self.is_global_zero else self.mute_tqdm
                                        for batch_idx, batch in enumerate(
                                                tqdm(dataloader, position=self.global_rank,
                                                     pre=current, desc=f"training #{self.global_rank + 1:01d}", unit=f"x{dataloader.batch_size}")):
                                            batch = to_tensor_batch(batch, input_keys=self.tokenizer.model_input_names)
                                            self.optimizer.zero_grad()
                                            output = self.each_step(batch, batch_idx, input_keys=self.tokenizer.model_input_names)
                                            outputs.append(output)
                                            logs["step"] += 1
                                            logs["epoch"] += epoch_per_step
                                            logs["learning_rate"] = self.get_learning_rate()
                                            self.backward(output['loss'])
                                            self.optimizer.step()
                                    metrics[k] = self.outputs_to_metrics(outputs, timer=timer)
                                self.scheduler.step()
                            marker.mark_done("TRAIN", stage=current, state_table_file=stderr)

                            # METER
                            self.finetuning_model.eval()
                            with torch.no_grad():
                                for k in self.input_datasets.keys():
                                    if k not in self.dataloader or not self.dataloader[k]:
                                        continue
                                    if k not in self.state.predicting_splits or not self.state.predicting_splits[k]:
                                        continue
                                    outputs = []
                                    dataloader = self.dataloader[k]
                                    with JobTimer() as timer:
                                        tqdm = self.time_tqdm if self.is_global_zero else self.mute_tqdm
                                        for batch_idx, batch in enumerate(
                                                tqdm(dataloader, position=self.global_rank,
                                                     pre=current, desc=f"metering #{self.global_rank + 1:01d}", unit=f"x{dataloader.batch_size}")):
                                            batch = to_tensor_batch(batch, input_keys=self.tokenizer.model_input_names)
                                            output = self.each_step(batch, batch_idx, input_keys=self.tokenizer.model_input_names)
                                            outputs.append(output)
                                    metrics[k] = self.outputs_to_metrics(outputs, timer=timer)
                            with JobTimer(verbose=True):
                                for name, score in metrics.items():
                                    print(self.time_tqdm.to_desc(pre=current, desc=f"measured #{self.global_rank + 1:01d}") +
                                          f": {name:<5s} | {', '.join(f'{k}={score[k]:.4f}' for k in append_intersection(score.keys(), ['runtime']))}")
                            marker.mark_done("METER", stage=current, state_table_file=stderr)

                            # SAVE
                            if self.state.finetuned_sub:
                                logs["state_path"] = new_path(finetuned_files["state"])
                                logs["model_path"] = new_path(finetuned_files["model"], post=f'{logs["epoch"]:02.0f}e')
                                self.save(self.finetuning_model.state_dict(), filepath=logs["model_path"])
                                if self.is_global_zero:
                                    logs["record"].append({
                                        "step": logs["step"],
                                        "epoch": logs["epoch"],
                                        "metrics": metrics,
                                        "model_path": logs["model_path"] if logs['model_path'].exists() else None,
                                        "learning_rate": logs["learning_rate"],
                                    })
                                    self.state.records = logs["record"]
                                    save_attrs(self.state, file=logs["state_path"], keys=self.state.log_targets)
                                with JobTimer(verbose=True):
                                    if self.is_global_zero and logs["model_path"].exists():
                                        print(self.time_tqdm.to_desc(pre=current, desc=f"exported #{self.global_rank + 1:01d}") + f": model | {logs['model_path']}")
                                marker.mark_done("SAVE", stage=current, state_table_file=stderr)

    def configure_strategy(self):
        if self.state.strategy == "dp":
            return DataParallelStrategy()
        if self.state.strategy == "ddp":
            return DDPStrategy()
        if self.state.strategy == "deepspeed":
            return DeepSpeedStrategy(stage=2)
        return None

    def configure_loss(self):
        if self.state.loss_metric == "MSELoss":
            return nn.MSELoss()
        if self.state.loss_metric == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        raise ValueError(f"Undefined loss_metric: {self.state.loss_metric}")

    def configure_optimizer(self):
        # https://pytorch.org/docs/stable/optim.html
        if self.state.optimizer_type == "SGD":  # TODO: SGD(momentum)
            return optim.SGD(self.finetuning_model.parameters(), lr=self.state.learning_rate)
        if self.state.optimizer_type == "Adam":
            return optim.Adam(self.finetuning_model.parameters(), lr=self.state.learning_rate)
        if self.state.optimizer_type == "AdamW":
            return optim.AdamW(self.finetuning_model.parameters(), lr=self.state.learning_rate)
        if self.state.optimizer_type == "Adagrad":
            return optim.Adagrad(self.finetuning_model.parameters(), lr=self.state.learning_rate)
        if self.state.optimizer_type == "Adadelta":
            return optim.Adadelta(self.finetuning_model.parameters(), lr=self.state.learning_rate)
        if self.state.optimizer_type == "RMSprop":
            return optim.RMSprop(self.finetuning_model.parameters(), lr=self.state.learning_rate)
        raise ValueError(f"Undefined lr_scheduler_type: {self.state.lr_scheduler_type}")

    def configure_scheduler(self):
        # TODO: CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts 비교
        if self.state.lr_scheduler_type == "StepLR":
            return optim.lr_scheduler.StepLR(self.optimizer, gamma=self.state.lr_scheduler_gamma, step_size=self.state.lr_scheduler_step_size)
        if self.state.lr_scheduler_type == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.state.lr_scheduler_gamma)
        if self.state.lr_scheduler_type == "CyclicLR":
            return optim.lr_scheduler.CyclicLR(self.optimizer, gamma=self.state.lr_scheduler_gamma)
        if self.state.lr_scheduler_type == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.state.lr_scheduler_T_max)
        if self.state.lr_scheduler_type == "CosineAnnealingWarmRestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        raise ValueError(f"Undefined lr_scheduler_type: {self.state.lr_scheduler_type}")

    def forward(self, **x):
        return self.finetuning_model(**x)

    def each_step(self, batch: dict, batch_idx, input_keys) -> dict:
        x = {k: batch.pop(k) for k in input_keys}
        o = self.forward(**x)
        if isinstance(o, SequenceClassifierOutput):
            p = o.logits
        else:
            p = o
        if len(self.state.label_column.split('.')) == 2:
            major, minor = self.state.label_column.split('.')
            y = batch.pop(major).pop(minor)
        else:
            y = batch.pop(self.state.label_column)
        if self.state.loss_metric.startswith('MSELoss'):  # regression
            p = p.view(-1)
            y = y.float().view(-1)
        else:  # classification
            y = y.long().view(-1)
        loss = self.loss_metric(input=p, target=y)
        return {'y': y, 'p': p, 'loss': loss}

    def outputs_to_metrics(self, outputs, timer: Optional[JobTimer] = None) -> dict:
        score = dict()
        if timer:
            score['runtime'] = timer.td.total_seconds()
        ys = torch.cat([x['y'] for x in outputs]).detach().cpu().numpy()
        ps = torch.cat([x['p'] for x in outputs]).detach().cpu().numpy()
        score['loss'] = torch.stack([x['loss'] for x in outputs]).detach().cpu().numpy().mean().item()
        if len(ps.shape) > 1:  # classification
            ps = np.argmax(ps, axis=1)
        score.update(self.score_metric.compute(references=ys, predictions=ps))
        return score

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def load_tokenizer(self, verbose=True) -> None:
        with JobTimer(verbose=verbose, mute_logger="transformers.tokenization_utils_base"):
            if self.state.pretrained.type == 'morp':
                self.tokenizer = KorbertTokenizer.from_pretrained(self.state.pretrained.path, max_len=self.state.max_sequence_length, use_fast=False, do_lower_case=False, tokenize_chinese_chars=False)
            elif self.state.pretrained.type == 'bbpe':
                self.tokenizer = AutoTokenizer.from_pretrained(self.state.pretrained.path, max_len=self.state.max_sequence_length, use_fast=False, do_lower_case=False)
                self.tok_coder = ByteLevelBPETokenizer(str(self.state.pretrained.path / "vocab.json"), str(self.state.pretrained.path / "merges.txt"))
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.state.pretrained.path, max_len=self.state.max_sequence_length)
            if 'additive_tokens' in self.state and self.state.pretrained.additive_tokens:
                if self.state.pretrained.additive_tokens in additive_tokens_for_morp_tag:
                    self.tokenizer.add_tokens(additive_tokens_for_morp_tag[self.state.pretrained.additive_tokens])
                    if self.tok_coder:
                        self.tok_coder.add_tokens(additive_tokens_for_morp_tag[self.state.pretrained.additive_tokens])

    def prepare_dataloader(self) -> None:
        # setup dataloaders
        for k in self.input_datasets.keys():
            if k not in self.state.batch_size_splits or self.state.batch_size_splits[k] <= 0:
                continue
            self.dataloader[k] = self.setup_dataloaders(DataLoader(self.input_datasets[k], batch_size=self.state.batch_size_splits[k], shuffle=False))

    def prepare_datasets(self, verbose=True, draw_figure=False) -> None:
        # load datasets
        with JobTimer(verbose=verbose):
            datasets.utils.logging.tqdm = self.time_tqdm if verbose else self.mute_tqdm
            data_files_to_load = {k: str(v) for k, v in self.state.data_files.items()
                                  if v and k in self.state.dataloader_splits and self.state.dataloader_splits[k]}
            with MuteDatasetProgress():
                self.input_datasets: DatasetDict = load_dataset(path="json", field="data",
                                                                name=','.join(f'{k}={Path(v).parent.name}' for k, v in data_files_to_load.items()),
                                                                cache_dir=self.state.cached_home / self.state.data_name / self.state.data_part,
                                                                data_files=data_files_to_load, download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS)
            assert len(self.input_datasets.keys()) > 0
        with JobTimer(verbose=verbose, rb=1):
            self.check_datasets(name="raw_datasets")

        # shuffle datasets
        for k in self.input_datasets:
            if k not in self.state.to_shuffle_splits or not self.state.to_shuffle_splits[k]:
                continue
            self.input_datasets[k] = self.input_datasets[k].shuffle(seed=self.state.seed, load_from_cache_file=False)

        # take datasets to use
        for k in self.input_datasets:
            if k not in self.state.use_sample_splits or self.state.use_sample_splits[k] <= 0:
                continue
            num_taken_samples = limit_num_samples(self.state.use_sample_splits[k], num_max=len(self.input_datasets[k]))
            self.input_datasets[k] = Dataset.from_dict(self.input_datasets[k][: num_taken_samples])

        # load tokenizer
        self.load_tokenizer(verbose=verbose)

        # encode datasets
        with JobTimer(verbose=verbose, rb=1):
            self.input_datasets, counted_datasets = self.encode_datasets()
        with JobTimer(verbose=verbose, rb=1):
            self.check_datasets(name="encoded_datasets")
        if counted_datasets:
            with JobTimer(verbose=verbose):
                self.display_counts(counted_datasets, verbose=verbose, figure=draw_figure)

        # sample examples
        if 'num_check_samples' in self.state and self.state.num_check_samples >= 1:
            first_split = list(self.input_datasets.keys())[0]
            first_dataset = self.input_datasets[first_split]
            sampled_example_idxs = self.rand.sample(range(first_dataset.num_rows), k=min(self.state.num_check_samples, first_dataset.num_rows))
            self.sample_dataset = first_dataset.select(sampled_example_idxs)

    def encode_datasets(self) -> tuple[DatasetDict, DatasetDict]:
        encoded_datasets = DatasetDict()
        counted_datasets = DatasetDict()
        for split, dataset in self.input_datasets.items():
            current = f"({split:<5s})"
            dataset: Dataset = dataset
            dataset_path = None if not self.state.cached_home or not self.state.dataset_home else (
                    Path(self.state.cached_home) /
                    Path(self.state.data_files[split]).with_suffix('').relative_to(self.state.dataset_home) /
                    '@'.join([
                        f"by-{self.state.pretrained.path.name}@vocab={len(self.tokenizer)}",
                        f"{'@'.join(f'{x.colname}={x.max_tokens}' for x in (self.state.input_text1, self.state.input_text2) if x and x.type and x.colname)}",
                        f"max_seq={self.state.max_sequence_length}",
                        f"row={dataset.num_rows}",
                    ])
            )
            encoded_dataset_path = None if not dataset_path else dataset_path.with_name(f"encoded-{dataset_path.name}")
            counted_dataset_path = None if not dataset_path else dataset_path.with_name(f"counted-{dataset_path.name}")
            if dataset_path and encoded_dataset_path and encoded_dataset_path.exists():
                encoded_datasets[split] = Dataset.load_from_disk(str(encoded_dataset_path))
                print(self.time_tqdm.to_desc(pre=current, desc=f"imported from") + f": {encoded_dataset_path}")
                if counted_dataset_path and counted_dataset_path.exists():
                    counted_datasets[split] = Dataset.load_from_disk(str(counted_dataset_path))
                    print(self.time_tqdm.to_desc(pre=current, desc=f"imported from") + f": {counted_dataset_path}")
            else:
                encode_batch_size = max(1, self.state.encode_batch_size)
                dataset = dataset.map(
                    self.count_input_text_batch, batched=True, batch_size=encode_batch_size,
                    desc=f"{current} {f'counting #{self.global_rank + 1:01d}':>{self.time_tqdm.desc_size}s}"
                )
                counted_datasets[split] = dataset
                encoded_datasets[split] = dataset.filter(
                    self.filter_input_text_batch, batched=True, batch_size=encode_batch_size
                ).map(
                    self.encode_example_batch, batched=True, batch_size=encode_batch_size,
                    desc=f"{current} {f'encoding #{self.global_rank + 1:01d}':>{self.time_tqdm.desc_size}s}"
                )
                if self.save_cache and self.is_global_zero and dataset_path and encoded_dataset_path:
                    encoded_datasets[split].save_to_disk(str(encoded_dataset_path))
                    print(self.time_tqdm.to_desc(pre=current, desc=f"exported to") + f": {encoded_dataset_path}")
                    if counted_dataset_path:
                        counted_datasets[split].save_to_disk(str(counted_dataset_path))
                        print(self.time_tqdm.to_desc(pre=current, desc=f"exported to") + f": {counted_dataset_path}")
        return encoded_datasets, counted_datasets

    def to_batch_text_pair(self, batch_text: Dataset | LazyBatch) -> list[list[TextInput]]:
        batch_text_pair = list()
        if self.state.input_text1 and self.state.input_text1.type and self.state.input_text1.colname:
            batch_text_pair.append(batch_text[self.state.input_text1.colname])
        if self.state.input_text2 and self.state.input_text2.type and self.state.input_text2.colname:
            batch_text_pair.append(batch_text[self.state.input_text2.colname])
        return batch_text_pair

    def count_input_text_batch(self, batch_text: Dataset | LazyBatch) -> Dataset | LazyBatch:
        for input_text in (self.state.input_text1, self.state.input_text2):
            if input_text and input_text.type and input_text.colname:
                # apply to_morphemes() to morp analized text
                if input_text.type == "morp" and f"{input_text.colname}_origin" not in batch_text:
                    batch_text[f"{input_text.colname}_origin"] = [x for x in batch_text[input_text.colname]]
                    batch_text[input_text.colname] = [to_morphemes(x) for x in batch_text[input_text.colname]]
                if input_text.max_tokens > 0 and f"length-{input_text.colname}" not in batch_text:
                    batch_seq: BatchEncoding = self.tokenizer.__call__(batch_text[input_text.colname],
                                                                       return_length=True, verbose=False,
                                                                       truncation=TruncationStrategy.DO_NOT_TRUNCATE,
                                                                       padding=PaddingStrategy.DO_NOT_PAD)
                    batch_text[f"length-{input_text.colname}"] = batch_seq['length']
        if "length-text_pair" not in batch_text:
            batch_text_pair: list[list[TextInput]] = self.to_batch_text_pair(batch_text)
            batch_seq: BatchEncoding = self.tokenizer.__call__(*batch_text_pair,
                                                               return_length=True, verbose=False,
                                                               truncation=TruncationStrategy.DO_NOT_TRUNCATE,
                                                               padding=PaddingStrategy.DO_NOT_PAD)
            batch_text["length-text_pair"] = batch_seq['length']
        return batch_text

    def encode_example_batch(self, batch_text: Dataset | LazyBatch) -> BatchEncoding:
        batch_text_pair: list[list[TextInput]] = self.to_batch_text_pair(batch_text)
        batch_seq: BatchEncoding = self.tokenizer.__call__(*batch_text_pair,
                                                           return_length=True, verbose=True,
                                                           max_length=self.state.max_sequence_length,
                                                           truncation=self.state.truncation_strategy,
                                                           padding=self.state.padding_strategy)
        return batch_seq

    def filter_input_text_batch(self, batch_text: Dataset | LazyBatch) -> list[bool]:
        passed = list()
        for input_text in (self.state.input_text1, self.state.input_text2):
            if input_text and input_text.type and input_text.colname and input_text.max_tokens > 0:
                passed.append([x <= input_text.max_tokens for x in batch_text[f"length-{input_text.colname}"]])
        return [all(x) for x in zip(*passed)]

    def decode_by_coder(self, token_ids):
        if self.tok_coder:
            return [no_space(no_replacement(no_nonprintable(self.tok_coder.decode([x])))) for x in token_ids]
        else:
            return None

    def display_counts(self, counted_datasets, show_min=5, show_max=15, verbose=True, figure=True) -> None:
        for split, dataset in counted_datasets.items():
            current = f"({split:<5s})"
            dataset: Dataset = dataset
            encoded_lengths = {
                x[len('length-'):]: dataset[x]
                for x in dataset.column_names if x.startswith('length-')
            }
            if verbose and figure:
                display_histogram(encoded_lengths, title=f"sequence lengths in {split} dataset")
            with JobTimer(verbose=verbose, rb=1):
                for text_field in encoded_lengths:
                    seq_lens = sorted(Counter(encoded_lengths[text_field]).items())
                    print(f"{self.time_tqdm.to_desc(pre=current, desc=f'#({text_field})')}: "
                          f"{', '.join([f'{k}({v})' for k, v in seq_lens[:show_min]])}, ..., {', '.join([f'{k}({v})' for k, v in seq_lens[-show_max:]])}")

    def check_datasets(self, name) -> None:
        for split, dataset in self.input_datasets.items():
            current = f"({split:<5s})"
            dataset: Dataset = dataset
            for f, v in dataset.features.items():
                assert isinstance(v, (datasets.Sequence, datasets.Value, dict)), f"feature({f}) is not {datasets.Sequence.__name__}, {datasets.Value.__name__} or {dict.__name__}"
                if isinstance(v, dict):
                    for f2, v2 in v.items():
                        assert isinstance(v2, datasets.Value), f"feature({f2}) is not {datasets.Value.__name__}"
            feature_specs = []
            for f, v in dataset.features.items():
                if isinstance(v, dict):
                    feature_specs.append(', '.join(f'{f}.{f2}[{v2.dtype}]' for f2, v2 in v.items()))
                else:
                    feature_specs.append(f'{f}[{v.dtype}]')
            print(f"{self.time_tqdm.to_desc(pre=current, desc=f'{name}')}: {dataset.num_rows:11,d} samples | {', '.join(feature_specs)}")

    def check_tokenizer(self, sample=True, num_show_tokens=50) -> None:
        print(f"- {'pretrained_tokenizer':30s} =", ' | '.join([
            f"{type(self.tokenizer).__qualname__}",
            f"inputs=({', '.join(self.tokenizer.model_input_names)})",
            f"max_length={self.tokenizer.model_max_length}",
            f"#vocab1={self.tokenizer.vocab_size}",
            f"#vocab2={len(self.tokenizer)}",
            f"#added_vocab={len(self.tokenizer) - self.tokenizer.vocab_size}",
        ]))
        if sample and self.sample_dataset:
            encoded_batch = self.encode_example_batch(self.sample_dataset)
            if self.sample_dataset:
                for eid, example in enumerate(self.sample_dataset):
                    with JobTimer(verbose=self.is_global_zero, rt=1):
                        input_text = [example[x.colname] for x in (self.state.input_text1, self.state.input_text2) if x and x.type and x.colname]
                        print(f"- {f'encoded[{eid}].input_txt':30s} = {input_text}")
                        for column_name in encoded_batch.keys():
                            if column_name == 'input_ids':
                                ids = encoded_batch[column_name][eid]
                                tks = self.decode_by_coder(ids) if self.tok_coder else self.tokenizer.convert_ids_to_tokens(ids)
                                print(f"- {f'encoded[{eid}].input_tks':30s}"
                                      f" = ({'x'.join(str(x) for x in list(torch.tensor(encoded_batch[column_name][eid]).size()))})"
                                      f" | {' '.join(str(x) for x in tks[: num_show_tokens])} ... {' '.join(str(x) for x in tks[-10:])}")
                                print(f"- {f'encoded[{eid}].{column_name}':30s}"
                                      f" = ({'x'.join(str(x) for x in list(torch.tensor(encoded_batch[column_name][eid]).size()))})"
                                      f" | {' '.join(str(x) for x in ids[: num_show_tokens])} ... {' '.join(str(x) for x in ids[-10:])}")

    def check_pretrained(self, sample=True, num_show_tokens=50) -> None:
        pretrained_model = self.finetuning_model.model.base_model
        print(f"- {'pretrained_model':30s} =", ' | '.join([
            f"{type(pretrained_model).__qualname__}",
            f"#attention_heads={pretrained_model.config.num_attention_heads}",
            f"#hidden_layers={pretrained_model.config.num_hidden_layers}",
            f"#embedding={pretrained_model.config.hidden_size}",
            f"#vocab={pretrained_model.config.vocab_size}",
        ]))
        if sample and self.sample_dataset:
            encoded_batch = self.encode_example_batch(self.sample_dataset)
            forwarded = pretrained_model.forward(
                *[torch.tensor(encoded_batch[input_name]) for input_name in self.tokenizer.model_input_names]
            )
            last_hidden = forwarded.last_hidden_state
            with JobTimer(verbose=self.is_global_zero, rt=1):
                sample_eid = 0
                for column_name in self.tokenizer.model_input_names:
                    ids = encoded_batch[column_name][sample_eid]
                    print(f"- {f'encoded.{column_name}':30s}"
                          f" = ({'x'.join(str(x) for x in list(torch.tensor(encoded_batch[column_name]).size()))})"
                          f" | {' '.join(str(x) for x in ids[: num_show_tokens])} ... {' '.join(str(x) for x in ids[-10:])}")
                print(f"- {'forwarded.hidden_output':30s} = ({'x'.join(str(x) for x in list(last_hidden.size()))}) | {last_hidden[0]}")
