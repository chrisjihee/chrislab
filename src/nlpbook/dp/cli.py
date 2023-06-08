import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightning as L
import lightning.pytorch as pl
import nlpbook
import torch
from chrisbase.io import JobTimer, pop_keys, err_hr, out_hr
from chrislab.common.util import time_tqdm_cls, mute_tqdm_cls
from flask import Flask
from klue_baseline.metrics.functional import klue_ner_entity_macro_f1, klue_ner_char_macro_f1
from nlpbook import new_set_logger
from nlpbook.arguments import TrainerArguments, ServerArguments, TesterArguments, RuntimeChecking
from nlpbook.metrics import accuracy
from nlpbook.dp.corpus import DPCorpus, DPDataset, dp_encoded_examples_to_batch
from nlpbook.ner.task import NERTask
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoConfig, AutoModelForTokenClassification, BertForTokenClassification, CharSpan
from transformers.modeling_outputs import TokenClassifierOutput
from typer import Typer

app = Typer()
logger = logging.getLogger("chrislab")
term_pattern = re.compile(re.escape("{") + "(.+?)(:.+?)?" + re.escape("}"))


@app.command()
def fabric_train(args_file: Path | str):
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args: TrainerArguments = TrainerArguments.from_json(args_file.read_text()).show()
    new_set_logger()
    L.seed_everything(args.learning.seed)

    with JobTimer(f"chrialab.nlpbook.dp fabric_train {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        # Data
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        corpus = DPCorpus(args)
        train_dataset = DPDataset("train", args=args, corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.batch_size,
                                      collate_fn=dp_encoded_examples_to_batch,
                                      drop_last=True)
        logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
        logger.info(f"Created train_dataloader loading {len(train_dataloader)} batches")
        args.output.epoch_per_step = 1 / len(train_dataloader)
        err_hr(c='-')
        valid_dataset = DPDataset("valid", args=args, corpus=corpus, tokenizer=tokenizer)
        valid_dataloader = DataLoader(valid_dataset,
                                      sampler=SequentialSampler(valid_dataset),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.batch_size,
                                      collate_fn=dp_encoded_examples_to_batch,
                                      drop_last=True)
        logger.info(f"Created valid_dataset providing {len(valid_dataset)} examples")
        logger.info(f"Created valid_dataloader loading {len(valid_dataloader)} batches")
        err_hr(c='-')

        # Model
        pretrained_model_config = AutoConfig.from_pretrained(
            args.model.pretrained,
            num_labels=corpus.num_labels
        )
        model = AutoModelForTokenClassification.from_pretrained(
            args.model.pretrained,
            config=pretrained_model_config
        )
        err_hr(c='-')

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning.speed)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # Fabric
        with RuntimeChecking(args.setup_csv_out()):
            torch.set_float32_matmul_precision('high')
            fabric = L.Fabric(loggers=args.output.csv_out)
            fabric.setup(model, optimizer)
            train_dataloader, valid_dataloader = fabric.setup_dataloaders(train_dataloader, valid_dataloader)
            train_with_fabric(fabric, args, model, optimizer, scheduler, train_dataloader, valid_dataloader, valid_dataset)


def train_with_fabric(fabric: L.Fabric, args: TrainerArguments,
                      model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                      train_dataloader: DataLoader, valid_dataloader: DataLoader, valid_dataset: DPDataset):
    time_tqdm = time_tqdm_cls(bar_size=20, desc_size=8, file=sys.stdout)
    mute_tqdm = mute_tqdm_cls()
    val_interval: float = args.learning.validating_on * len(train_dataloader) if args.learning.validating_on <= 1.0 else args.learning.validating_on
    sorted_checkpoints: List[Tuple[float, Path]] = []
    sorting_reverse: bool = not args.learning.keeping_by.split()[0].lower().startswith("min")
    sorting_metric: str = args.learning.keeping_by.split()[-1]
    metrics: Dict[str, Any] = {}
    args.output.global_step = 0
    args.output.global_epoch = 0.0
