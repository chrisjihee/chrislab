import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoConfig, AutoModel, BertConfig, BertModel, RobertaConfig, RobertaModel, PreTrainedModel, PretrainedConfig
from typer import Typer

from chrisbase.data import RuntimeChecking
from chrisbase.io import JobTimer, pop_keys, err_hr
from chrislab.common.util import time_tqdm_cls, mute_tqdm_cls
from nlpbook import save_checkpoint
from nlpbook.arguments import TrainerArguments
from nlpbook.dp.corpus import DPCorpus, DPDataset
from nlpbook.dp.model import ModelForDependencyParsing
from nlpbook.metrics import DPResult

app = Typer()
logger = logging.getLogger(__name__)
term_pattern = re.compile(re.escape("{") + "(.+?)(:.+?)?" + re.escape("}"))


@app.command()
def fabric_train(args_file: Path | str):
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args: TrainerArguments = TrainerArguments.from_json(args_file.read_text()).info_args()
    L.seed_everything(args.learning.random_seed)

    with JobTimer(f"chrialab.nlpbook.dp fabric_train {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        # Data
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        corpus = DPCorpus(args)
        train_dataset = DPDataset("train", corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset,
                                      # sampler=SequentialSampler(train_dataset),  # TODO: temporary
                                      sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.train_batch,
                                      collate_fn=corpus.encoded_examples_to_batch,
                                      # drop_last=True,
                                      drop_last=False,  # TODO: temporary
                                      )
        logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
        logger.info(f"Created train_dataloader loading {len(train_dataloader)} batches")
        args.prog.epoch_per_step = 1 / len(train_dataloader)
        err_hr(c='-')
        valid_dataset = DPDataset("valid", corpus=corpus, tokenizer=tokenizer)
        valid_dataloader = DataLoader(valid_dataset,
                                      sampler=SequentialSampler(valid_dataset),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.infer_batch,
                                      collate_fn=corpus.encoded_examples_to_batch,
                                      # drop_last=True,
                                      drop_last=False,  # TODO: temporary
                                      )
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
        model = ModelForDependencyParsing(args, corpus, pretrained_model)
        err_hr(c='-')

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # Fabric
        with RuntimeChecking(args.configure_csv_logger()):
            torch.set_float32_matmul_precision('high')
            fabric = L.Fabric(
                accelerator=args.hardware.accelerator,
                precision=args.hardware.precision,
                strategy=args.hardware.strategy,
                devices=args.hardware.devices,
                loggers=args.prog.csv_logger,
            )
            fabric.setup(model, optimizer)
            train_dataloader, valid_dataloader = fabric.setup_dataloaders(train_dataloader, valid_dataloader)
            train_with_fabric(fabric, args, model, optimizer, scheduler, train_dataloader, valid_dataloader, valid_dataset)


def train_with_fabric(fabric: L.Fabric, args: TrainerArguments, model: ModelForDependencyParsing,
                      optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                      train_dataloader: DataLoader, valid_dataloader: DataLoader, valid_dataset: DPDataset):
    time_tqdm = time_tqdm_cls(bar_size=20, desc_size=8, file=sys.stdout)
    mute_tqdm = mute_tqdm_cls()
    val_interval: float = args.learning.validate_on * len(train_dataloader) if args.learning.validate_on <= 1.0 else args.learning.validate_on
    sorted_checkpoints: List[Tuple[float, Path]] = []
    sorting_reverse: bool = not args.learning.saving_mode.split()[0].lower().startswith("min")
    sorting_metric: str = args.learning.saving_mode.split()[-1]
    metric_values: Dict[str, Any] = {}
    args.prog.global_step = 0
    args.prog.global_epoch = 0.0
    for epoch in range(args.learning.num_epochs):
        epoch_info = f"(Epoch {epoch + 1:02d})"
        metric_values["epoch"] = round(args.prog.global_epoch, 4)
        metric_values["trained_rate"] = round(args.prog.global_epoch, 4) / args.learning.num_epochs
        metric_values["lr"] = optimizer.param_groups[0]['lr']
        epoch_tqdm = time_tqdm if fabric.is_global_zero else mute_tqdm
        assert len(train_dataloader) > 0
        for batch_idx, batch in enumerate(epoch_tqdm(train_dataloader, position=fabric.global_rank, pre=epoch_info, desc="training", unit="batch")):
            model.train()
            args.prog.global_step += 1
            args.prog.global_epoch += args.prog.epoch_per_step
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

            metric_values["epoch"] = round(args.prog.global_epoch, 4)
            metric_values["trained_rate"] = round(args.prog.global_epoch, 4) / args.learning.num_epochs
            metric_values["loss"] = loss.item()

            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, clip_val=0.25)
            optimizer.step()
            optimizer.zero_grad()

            model.eval()
            if batch_idx + 1 == len(train_dataloader) or (batch_idx + 1) % val_interval < 1:
                validate(fabric, args, model, valid_dataloader, valid_dataset,
                         metric_values=metric_values, print_result=args.learning.tag_format_on_validate is not None)
                sorted_checkpoints = save_checkpoint(fabric, args, metric_values, model, optimizer,
                                                     sorted_checkpoints, sorting_reverse, sorting_metric)
            fabric.log_dict(step=args.prog.global_step, metrics=metric_values)


@torch.no_grad()
def validate(fabric: L.Fabric, args: TrainerArguments, model: ModelForDependencyParsing,
             valid_dataloader: DataLoader, valid_dataset: DPDataset,
             metric_values: Dict[str, Any], print_result: bool = True):
    metric_values["val_loss"] = torch.zeros(len(valid_dataloader))
    all_preds = []
    all_labels = []
    assert len(valid_dataloader) > 0
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
            is_training=False,
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

        metric_values["val_loss"][batch_idx] = loss

        # predict arc and its type
        pred_heads: torch.Tensor = torch.argmax(out_arc, dim=2)
        pred_types: torch.Tensor = torch.argmax(out_type, dim=2)
        preds = DPResult(pred_heads, pred_types)
        labels = DPResult(batch["head_ids"], batch["type_ids"])
        all_preds.append(preds)
        all_labels.append(labels)

    metric_values["val_loss"] = metric_values["val_loss"].mean().item()
    assert len(all_preds) > 0 and len(all_labels) and len(all_preds) == len(all_labels)
    for k, metric_tool in model.metric_tools.items():
        metric_tool(all_preds, all_labels)
        metric_values[f"val_{k}"] = metric_tool.compute()

    if print_result:
        terms = [m.group(1) for m in term_pattern.finditer(args.learning.tag_format_on_validate)]
        terms = {term: metric_values[term] for term in terms}
        fabric.print(' | ' + args.learning.tag_format_on_validate.format(**terms))
