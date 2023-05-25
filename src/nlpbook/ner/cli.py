import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from flask import Flask
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typer import Typer

import lightning as L
import nlpbook
from chrisbase.io import JobTimer, pop_keys, err_hr, out_hr
from chrislab.common.util import time_tqdm_cls, mute_tqdm_cls
from klue_baseline.metrics.functional import klue_ner_entity_macro_f1, klue_ner_char_macro_f1
from nlpbook.arguments import TrainerArguments, ServerArguments, TesterArguments, RuntimeChecking
from nlpbook.metrics import accuracy
from nlpbook.ner.corpus import NERCorpus, NERDataset, encoded_examples_to_batch, NEREncodedExample
from nlpbook.ner.task import NERTask
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoConfig, AutoModelForTokenClassification, BertForTokenClassification, CharSpan
from transformers.modeling_outputs import TokenClassifierOutput

app = Typer()
logger = logging.getLogger("chrislab")
term_pattern = re.compile(re.escape("{") + "(.+?)(:.+?)?" + re.escape("}"))


def new_set_logger(level=logging.INFO):
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(levelname)s\t%(name)s\t%(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(level)


def new_set_logger2(level=logging.INFO, filename="running.log", fmt="%(levelname)s\t%(name)s\t%(message)s"):
    from chrisbase.io import sys_stderr, sys_stdout
    stream_handler = logging.StreamHandler(stream=sys_stderr)
    file_handler = logging.FileHandler(filename=filename, mode="w", encoding="utf-8")

    stream_handler.setFormatter(logging.Formatter(fmt=fmt))
    file_handler.setFormatter(logging.Formatter(fmt=fmt))

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.setLevel(level)


class FabricTrainer(L.Fabric):
    def __init__(self, args: TrainerArguments):
        # Data
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        corpus = NERCorpus(args)
        train_dataset = NERDataset("train", args=args, corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.batch_size,
                                      collate_fn=encoded_examples_to_batch,
                                      drop_last=True)
        logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
        logger.info(f"Created train_dataloader loading {len(train_dataloader)} batches")
        args.output.epoch_per_step = 1 / len(train_dataloader)
        err_hr(c='-')
        valid_dataset = NERDataset("valid", args=args, corpus=corpus, tokenizer=tokenizer)
        valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.batch_size,
                                      collate_fn=encoded_examples_to_batch,
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
        args.setup_csv_out()
        super().__init__(
            accelerator=args.hardware.accelerator,
            precision=args.hardware.precision,
            strategy=args.hardware.strategy,
            devices=args.hardware.devices,
            loggers=args.output.csv_out
        )
        self.setup(model, optimizer)
        train_dataloader, valid_dataloader = self.setup_dataloaders(train_dataloader, valid_dataloader)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.args = args
        train_with_fabric(self, args, model, optimizer, scheduler, train_dataloader, valid_dataloader, valid_dataset)

    def run(self):
        with RuntimeChecking():
            print(f"self.args.output.csv_out.log_dir={self.args.output.csv_out.log_dir}")
            print("RUNTIME!!")
            logger.info("RUNTIME (2)")


@app.command()
def new_train(args_file: Path | str):
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args: TrainerArguments = TrainerArguments.from_json(args_file.read_text()).show()
    new_set_logger()
    L.seed_everything(args.learning.seed)

    with JobTimer(f"chrialab.nlpbook.ner new_train {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        # Data
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        corpus = NERCorpus(args)
        train_dataset = NERDataset("train", args=args, corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.batch_size,
                                      collate_fn=encoded_examples_to_batch,
                                      drop_last=True)
        logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
        logger.info(f"Created train_dataloader loading {len(train_dataloader)} batches")
        args.output.epoch_per_step = 1 / len(train_dataloader)
        err_hr(c='-')
        valid_dataset = NERDataset("valid", args=args, corpus=corpus, tokenizer=tokenizer)
        valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.batch_size,
                                      collate_fn=encoded_examples_to_batch,
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


# https://lightning.ai/docs/fabric/stable/guide/checkpoint.html
def save_checkpoint(fabric, args, metrics, model, optimizer,
                    sorted_checkpoints: List[Tuple[float, Path]], sorting_reverse, sorting_metric):
    checkpoint_state = {"model": model, "optimizer": optimizer, "args": args}
    terms = [m.group(1) for m in term_pattern.finditer(args.model.finetuning_name)]
    terms = {term: metrics[term] for term in terms}
    checkpoint_stem = args.model.finetuning_name.format(**terms)
    checkpoint_path: Path = (args.output.dir_path / "model.out").with_stem(checkpoint_stem).with_suffix(".ckpt")

    sorted_checkpoints.append((metrics[sorting_metric], checkpoint_path))
    sorted_checkpoints.sort(key=lambda x: x[0], reverse=sorting_reverse)
    for _, path in sorted_checkpoints[args.learning.num_keeping:]:
        path.unlink(missing_ok=True)
    sorted_checkpoints = [(value, path) for value, path in sorted_checkpoints[:args.learning.num_keeping] if path.exists()]
    if len(sorted_checkpoints) < args.learning.num_keeping:
        fabric.save(checkpoint_path, checkpoint_state)
        sorted_checkpoints.append((metrics[sorting_metric], checkpoint_path))
        sorted_checkpoints.sort(key=lambda x: x[0], reverse=sorting_reverse)
    return sorted_checkpoints


def train_with_fabric(fabric: L.Fabric, args: TrainerArguments,
                      model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                      train_dataloader: DataLoader, valid_dataloader: DataLoader, valid_dataset: NERDataset):
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
        metrics["lr"] = optimizer.param_groups[0]['lr']
        epoch_tqdm = time_tqdm if fabric.is_global_zero else mute_tqdm
        for batch_idx, batch in enumerate(epoch_tqdm(train_dataloader, position=fabric.global_rank, pre=epoch_info,
                                                     desc=f"training", unit=f"x{train_dataloader.batch_size}")):
            args.output.global_step += 1
            args.output.global_epoch += args.output.epoch_per_step
            batch: Dict[str, torch.Tensor] = pop_keys(batch, "example_ids")
            outputs: TokenClassifierOutput = model(**batch)
            labels: torch.Tensor = batch["labels"]
            preds: torch.Tensor = outputs.logits.argmax(dim=-1)
            acc: torch.Tensor = accuracy(preds, labels, ignore_index=0)
            metrics["epoch"] = round(args.output.global_epoch, 4)
            metrics["loss"] = outputs.loss.item()
            metrics["acc"] = acc.item()
            fabric.backward(outputs.loss)
            fabric.clip_gradients(model, optimizer, clip_val=0.25)
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx + 1 == len(train_dataloader) or (batch_idx + 1) % val_interval < 1:
                validate(fabric, args, model, valid_dataloader, valid_dataset, metrics=metrics, print_result=args.learning.validating_fmt is not None)
                sorted_checkpoints = save_checkpoint(fabric, args, metrics, model, optimizer,
                                                     sorted_checkpoints, sorting_reverse, sorting_metric)
            fabric.log_dict(step=args.output.global_step, metrics=metrics)
        scheduler.step()
        metrics["lr"] = optimizer.param_groups[0]['lr']
        if epoch + 1 < args.learning.epochs:
            out_hr('-')


def label_to_char_labels(label, num_char):
    for i in range(num_char):
        if i > 0 and ("-" in label):
            yield "I-" + label.split("-", maxsplit=1)[-1]
        else:
            yield label


@torch.no_grad()
def validate(fabric: L.Fabric, args: TrainerArguments, model: torch.nn.Module,
             valid_dataloader: DataLoader, valid_dataset: NERDataset,
             metrics: Dict[str, Any], print_result: bool = True):
    model.eval()
    metrics["val_loss"] = torch.zeros(len(valid_dataloader))
    # metrics["val_acc"] = torch.zeros(len(valid_dataloader))
    whole_char_label_pairs: List[Tuple[int, int]] = []
    for batch_idx, batch in enumerate(valid_dataloader):
        example_ids: torch.Tensor = batch.pop("example_ids")
        outputs: TokenClassifierOutput = model(**batch)
        # labels: torch.Tensor = batch["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        # acc: torch.Tensor = accuracy(preds, labels, ignore_index=0)
        for example_id, pred_ids in zip(example_ids.tolist(), preds.tolist()):
            pred_labels = [valid_dataset.id_to_label(x) for x in pred_ids]
            encoded_example: NEREncodedExample = valid_dataset[example_id]
            offset_to_label: Dict[int, str] = encoded_example.raw.get_offset_label_dict()
            char_label_pairs: List[Tuple[str | None, str | None]] = [(None, None)] * len(encoded_example.raw.character_list)
            for token_id in range(args.model.max_seq_length):
                token_span: CharSpan = encoded_example.encoded.token_to_chars(token_id)
                if token_span:
                    char_pred_tags = label_to_char_labels(pred_labels[token_id], token_span.end - token_span.start)
                    for offset, char_pred_tag in zip(range(token_span.start, token_span.end), char_pred_tags):
                        char_label_pairs[offset] = (offset_to_label[offset], char_pred_tag)
            whole_char_label_pairs.extend([(valid_dataset.label_to_id(y), valid_dataset.label_to_id(p))
                                           for y, p in char_label_pairs if y and p])
        metrics["val_loss"][batch_idx] = outputs.loss
        # metrics["val_acc"][batch_idx] = acc
    metrics["val_loss"] = metrics["val_loss"].mean().item()
    # metrics["val_acc"] = metrics["val_acc"].mean().item()
    char_preds, char_labels = ([p for (y, p) in whole_char_label_pairs],
                               [y for (y, p) in whole_char_label_pairs])
    metrics["val_f1c"] = klue_ner_char_macro_f1(preds=char_preds, labels=char_labels, label_list=valid_dataset.get_labels())
    metrics["val_f1e"] = klue_ner_entity_macro_f1(preds=char_preds, labels=char_labels, label_list=valid_dataset.get_labels())
    if print_result:
        terms = [m.group(1) for m in term_pattern.finditer(args.learning.validating_fmt)]
        terms = {term: metrics[term] for term in terms}
        fabric.print(' | ' + args.learning.validating_fmt.format(**terms))


@app.command()
def train(args_file: Path | str):
    nlpbook.set_logger()
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args: TrainerArguments = TrainerArguments.from_json(args_file.read_text()).show()
    nlpbook.set_seed(args)

    with JobTimer(f"chrialab.nlpbook.ner train {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        if args.data.redownload:
            nlpbook.download_downstream_dataset(args)
            err_hr(c='-')

        corpus = NERCorpus(args)
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        train_dataset = NERDataset("train", args=args, corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.batch_size,
                                      collate_fn=encoded_examples_to_batch,
                                      drop_last=False)
        err_hr(c='-')

        val_dataset = NERDataset("valid", args=args, corpus=corpus, tokenizer=tokenizer)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    num_workers=args.hardware.cpu_workers,
                                    batch_size=args.hardware.batch_size,
                                    collate_fn=encoded_examples_to_batch,
                                    drop_last=False)
        err_hr(c='-')

        pretrained_model_config = AutoConfig.from_pretrained(
            args.model.pretrained,
            num_labels=corpus.num_labels
        )
        model = AutoModelForTokenClassification.from_pretrained(
            args.model.pretrained,
            config=pretrained_model_config
        )
        err_hr(c='-')

        with RuntimeChecking(nlpbook.setup_csv_out(args)):
            torch.set_float32_matmul_precision('high')
            trainer: pl.Trainer = nlpbook.make_trainer(args)
            trainer.fit(NERTask(model=model, args=args, trainer=trainer, val_dataset=val_dataset,
                                total_steps=len(train_dataloader) * args.learning.epochs),
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)


@app.command()
def test(args_file: Path | str):
    nlpbook.set_logger()
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args = TesterArguments.from_json(args_file.read_text()).show()

    with JobTimer(f"chrialab.nlpbook.ner test {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        checkpoint_path = args.output.dir_path / args.model.finetuning_name
        assert checkpoint_path.exists(), f"No checkpoint file: {checkpoint_path}"
        logger.info(f"Using finetuned checkpoint file at {checkpoint_path}")
        err_hr(c='-')

        nlpbook.download_downstream_dataset(args)
        err_hr(c='-')

        corpus = NERCorpus(args)
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        test_dataset = NERDataset("test", args=args, corpus=corpus, tokenizer=tokenizer)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.hardware.batch_size,
                                     num_workers=args.hardware.cpu_workers,
                                     sampler=SequentialSampler(test_dataset),
                                     collate_fn=nlpbook.data_collator,
                                     drop_last=False)
        err_hr(c='-')

        pretrained_model_config = AutoConfig.from_pretrained(
            args.model.pretrained,
            num_labels=corpus.num_labels
        )
        model = AutoModelForTokenClassification.from_pretrained(
            args.model.pretrained,
            config=pretrained_model_config
        )
        err_hr(c='-')

        with RuntimeChecking(nlpbook.setup_csv_out(args)):
            torch.set_float32_matmul_precision('high')
            tester: pl.Trainer = nlpbook.make_tester(args)
            tester.test(NERTask(model, args, tester),
                        dataloaders=test_dataloader,
                        ckpt_path=checkpoint_path)


@app.command()
def serve(args_file: Path | str):
    nlpbook.set_logger()
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file file: {args_file}"
    args: ServerArguments = ServerArguments.from_json(args_file.read_text()).show()

    with JobTimer(f"chrialab.nlpbook serve_ner {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        checkpoint_path = args.output.dir_path / args.model.finetuning_name
        assert checkpoint_path.exists(), f"No checkpoint file: {checkpoint_path}"
        checkpoint: dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        logger.info(f"Using finetuned checkpoint file at {checkpoint_path}")
        err_hr(c='-')

        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        label_map_path: Path = args.output.dir_path / "label_map.txt"
        assert label_map_path.exists(), f"No downstream label file: {label_map_path}"
        labels = label_map_path.read_text().splitlines(keepends=False)
        id_to_label = {idx: label for idx, label in enumerate(labels)}

        pretrained_model_config = AutoConfig.from_pretrained(
            args.model.pretrained,
            num_labels=checkpoint['state_dict']['model.classifier.bias'].shape.numel(),
        )
        model = BertForTokenClassification(pretrained_model_config)
        model.load_state_dict({k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()})
        model.eval()
        err_hr(c='-')

        def inference_fn(sentence):
            inputs = tokenizer(
                [sentence],
                max_length=args.model.max_seq_length,
                padding="max_length",
                truncation=True,
            )
            with torch.no_grad():
                outputs: TokenClassifierOutput = model(**{k: torch.tensor(v) for k, v in inputs.items()})
                all_probs: Tensor = outputs.logits[0].softmax(dim=1)
                top_probs, top_preds = torch.topk(all_probs, dim=1, k=1)
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                top_labels = [id_to_label[pred[0].item()] for pred in top_preds]
                result = []
                for token, label, top_prob in zip(tokens, top_labels, top_probs):
                    if token in tokenizer.all_special_tokens:
                        continue
                    result.append({
                        "token": token,
                        "label": label,
                        "prob": f"{round(top_prob[0].item(), 4):.4f}",
                    })
            return {
                'sentence': sentence,
                'result': result,
            }

        with RuntimeChecking(nlpbook.setup_csv_out(args)):
            server: Flask = nlpbook.make_server(inference_fn,
                                                template_file="serve_ner.html",
                                                ngrok_home=args.env.working_path)
            server.run()
