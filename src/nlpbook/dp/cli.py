import torch.nn.functional as F
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
from nlpbook.dp.model import DPTransformer
from nlpbook.metrics import accuracy
from nlpbook.dp.corpus import DPCorpus, DPDataset  # , dp_encoded_examples_to_batch
from nlpbook.ner.task import NERTask
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoConfig, AutoModelForTokenClassification, BertForTokenClassification, RobertaForTokenClassification, CharSpan, AutoModel, BertConfig, BertModel, RobertaConfig, RobertaModel, PreTrainedModel, PretrainedConfig
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
                                      # sampler=SequentialSampler(train_dataset),  # TODO: temporary
                                      sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.batch_size,
                                      collate_fn=corpus.dp_encoded_examples_to_batch,
                                      drop_last=True,
                                      # drop_last=False,  # TODO: temporary
                                      )
        logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
        logger.info(f"Created train_dataloader loading {len(train_dataloader)} batches")
        args.output.epoch_per_step = 1 / len(train_dataloader)
        err_hr(c='-')
        valid_dataset = DPDataset("valid", args=args, corpus=corpus, tokenizer=tokenizer)
        valid_dataloader = DataLoader(valid_dataset,
                                      sampler=SequentialSampler(valid_dataset),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.batch_size,
                                      collate_fn=corpus.dp_encoded_examples_to_batch,
                                      drop_last=True)
        logger.info(f"Created valid_dataset providing {len(valid_dataset)} examples")
        logger.info(f"Created valid_dataloader loading {len(valid_dataloader)} batches")
        err_hr(c='-')

        # Model
        pretrained_model_config: PretrainedConfig | BertConfig | RobertaConfig = AutoConfig.from_pretrained(
            args.model.pretrained,
            num_labels=corpus.num_labels
        )
        pretrained_model: PreTrainedModel | BertModel | RobertaModel = AutoModel.from_pretrained(
            args.model.pretrained,
            config=pretrained_model_config
        )
        model = DPTransformer(args, corpus, pretrained_model)
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


def train_with_fabric(fabric: L.Fabric, args: TrainerArguments, model: DPTransformer,
                      optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
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
    for epoch in range(args.learning.epochs):
        epoch_info = f"(Epoch {epoch + 1:02d})"
        metrics["epoch"] = round(args.output.global_epoch, 4)
        metrics["trained_rate"] = round(args.output.global_epoch, 4) / args.learning.epochs
        metrics["lr"] = optimizer.param_groups[0]['lr']
        epoch_tqdm = time_tqdm if fabric.is_global_zero else mute_tqdm
        for batch_idx, batch in enumerate(epoch_tqdm(train_dataloader, position=fabric.global_rank, pre=epoch_info,
                                                     desc=f"training", unit=f"x{train_dataloader.batch_size}")):
            args.output.global_step += 1
            args.output.global_epoch += args.output.epoch_per_step
            batch: Dict[str, torch.Tensor] = pop_keys(batch, "example_ids")
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

            batch_size = batch["head_ids"].size()[0]
            batch_index = torch.arange(0, int(batch_size)).long()
            max_word_length = batch["max_word_length"].item()
            head_index = (
                torch.arange(0, max_word_length).view(max_word_length, 1).expand(max_word_length, batch_size).long()
            )

            # forward
            out_arc, out_type = model.forward(
                batch["bpe_head_mask"],
                batch["bpe_tail_mask"],
                batch["pos_ids"],
                batch["head_ids"],
                max_word_length,
                batch["mask_e"],
                batch["mask_d"],
                batch_index,
                **inputs,
            )

            # compute loss
            minus_inf = -1e8
            minus_mask_d = (1 - batch["mask_d"]) * minus_inf
            minus_mask_e = (1 - batch["mask_e"]) * minus_inf
            out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

            loss_arc = F.log_softmax(out_arc, dim=2)
            loss_type = F.log_softmax(out_type, dim=2)

            loss_arc = loss_arc * batch["mask_d"].unsqueeze(2) * batch["mask_e"].unsqueeze(1)
            loss_type = loss_type * batch["mask_d"].unsqueeze(2)
            num = batch["mask_d"].sum()

            loss_arc = loss_arc[batch_index, head_index, batch["head_ids"].data.t()].transpose(0, 1)
            loss_type = loss_type[batch_index, head_index, batch["type_ids"].data.t()].transpose(0, 1)
            loss_arc = -loss_arc.sum() / num
            loss_type = -loss_type.sum() / num
            loss = loss_arc + loss_type

            metrics["epoch"] = round(args.output.global_epoch, 4)
            metrics["trained_rate"] = round(args.output.global_epoch, 4) / args.learning.epochs
            metrics["loss"] = loss.item()

            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, clip_val=0.25)
            optimizer.step()
            optimizer.zero_grad()

            model.eval()
            if batch_idx + 1 == len(train_dataloader) or (batch_idx + 1) % val_interval < 1:
                validate(fabric, args, model, valid_dataloader, valid_dataset, metrics=metrics, print_result=args.learning.validating_fmt is not None)
            #     sorted_checkpoints = save_checkpoint(fabric, args, metrics, model, optimizer,
            #                                          sorted_checkpoints, sorting_reverse, sorting_metric)
            # fabric.log_dict(step=args.output.global_step, metrics=metrics)
            # model.train()


@torch.no_grad()
def validate(fabric: L.Fabric, args: TrainerArguments, model: DPTransformer,
             valid_dataloader: DataLoader, valid_dataset: DPDataset,
             metrics: Dict[str, Any], print_result: bool = True):
    metrics["val_loss"] = torch.zeros(len(valid_dataloader))
    for batch_idx, batch in enumerate(valid_dataloader):
        example_ids: torch.Tensor = batch.pop("example_ids")
        inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

        batch_size = batch["head_ids"].size()[0]
        batch_index = torch.arange(0, int(batch_size)).long()
        max_word_length = batch["max_word_length"].item()
        head_index = (
            torch.arange(0, max_word_length).view(max_word_length, 1).expand(max_word_length, batch_size).long()
        )

        # forward
        out_arc, out_type = model.forward(
            batch["bpe_head_mask"],
            batch["bpe_tail_mask"],
            batch["pos_ids"],
            batch["head_ids"],
            max_word_length,
            batch["mask_e"],
            batch["mask_d"],
            batch_index,
            **inputs,
        )

        # compute loss
        minus_inf = -1e8
        minus_mask_d = (1 - batch["mask_d"]) * minus_inf
        minus_mask_e = (1 - batch["mask_e"]) * minus_inf
        out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

        loss_arc = F.log_softmax(out_arc, dim=2)
        loss_type = F.log_softmax(out_type, dim=2)

        loss_arc = loss_arc * batch["mask_d"].unsqueeze(2) * batch["mask_e"].unsqueeze(1)
        loss_type = loss_type * batch["mask_d"].unsqueeze(2)
        num = batch["mask_d"].sum()

        loss_arc = loss_arc[batch_index, head_index, batch["head_ids"].data.t()].transpose(0, 1)
        loss_type = loss_type[batch_index, head_index, batch["type_ids"].data.t()].transpose(0, 1)
        loss_arc = -loss_arc.sum() / num
        loss_type = -loss_type.sum() / num
        loss = loss_arc + loss_type

        metrics["val_loss"][batch_idx] = loss
    metrics["val_loss"] = metrics["val_loss"].mean().item()
    if print_result:
        terms = [m.group(1) for m in term_pattern.finditer(args.learning.validating_fmt)]
        terms = {term: metrics[term] for term in terms}
        fabric.print(' | ' + args.learning.validating_fmt.format(**terms))
