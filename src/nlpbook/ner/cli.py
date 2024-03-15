import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightning as L
import pytorch_lightning as pl
import torch
from flask import Flask
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoConfig, AutoModelForTokenClassification, BertForTokenClassification, CharSpan
from transformers.modeling_outputs import TokenClassifierOutput
from typer import Typer

import nlpbook
from chrisbase.data import RuntimeChecking
from chrisbase.io import JobTimer, pop_keys, err_hr, out_hr
from chrislab.common.util import time_tqdm_cls, mute_tqdm_cls
from nlpbook import save_checkpoint, TERM_IN_NAME_FORMAT
from nlpbook.arguments import TrainerArguments, ServerArguments, TesterArguments
from nlpbook.metrics import accuracy, klue_ner_char_macro_f1, klue_ner_entity_macro_f1
from nlpbook.ner.corpus import NERCorpus, NERDataset, NEREncodedExample
from nlpbook.ner.task import NERTask

app = Typer()
logger = logging.getLogger(__name__)


@app.command()
def fabric_train(args_file: Path | str):
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args: TrainerArguments = TrainerArguments.from_json(args_file.read_text()).info_args()
    L.seed_everything(args.learning.random_seed)

    with JobTimer(f"chrialab.nlpbook.ner fabric_train {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        # Data
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        corpus = NERCorpus(args=args)
        train_dataset = NERDataset("train", corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.train_batch,
                                      collate_fn=corpus.encoded_examples_to_batch,
                                      drop_last=True)
        logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
        logger.info(f"Created train_dataloader loading {len(train_dataloader)} batches")
        args.prog.epoch_per_step = 1 / len(train_dataloader)
        err_hr(c='-')
        valid_dataset = NERDataset("valid", corpus=corpus, tokenizer=tokenizer)
        valid_dataloader = DataLoader(valid_dataset,
                                      sampler=SequentialSampler(valid_dataset),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.infer_batch,
                                      collate_fn=corpus.encoded_examples_to_batch,
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # Fabric
        with RuntimeChecking(args.configure_csv_logger()):
            torch.set_float32_matmul_precision('high')
            fabric = L.Fabric(loggers=args.prog.csv_logger)
            fabric.setup(model, optimizer)
            train_dataloader, valid_dataloader = fabric.setup_dataloaders(train_dataloader, valid_dataloader)
            train_with_fabric(fabric, args, model, optimizer, scheduler, train_dataloader, valid_dataloader, valid_dataset)


def train_with_fabric(fabric: L.Fabric, args: TrainerArguments, model: torch.nn.Module,
                      optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                      train_dataloader: DataLoader, valid_dataloader: DataLoader, valid_dataset: NERDataset):
    time_tqdm = time_tqdm_cls(bar_size=20, desc_size=8, file=sys.stdout)
    mute_tqdm = mute_tqdm_cls()
    val_interval: float = args.learning.validate_on * len(train_dataloader) if args.learning.validate_on <= 1.0 else args.learning.validate_on
    sorted_checkpoints: List[Tuple[float, Path]] = []
    sorting_reverse: bool = not args.learning.saving_mode.split()[0].lower().startswith("min")
    sorting_metric: str = args.learning.saving_mode.split()[-1]
    metrics: Dict[str, Any] = {}
    args.prog.global_step = 0
    args.prog.global_epoch = 0.0
    for epoch in range(args.learning.num_epochs):
        epoch_info = f"(Epoch {epoch + 1:02d})"
        metrics["epoch"] = round(args.prog.global_epoch, 4)
        metrics["trained_rate"] = round(args.prog.global_epoch, 4) / args.learning.num_epochs
        metrics["lr"] = optimizer.param_groups[0]['lr']
        epoch_tqdm = time_tqdm if fabric.is_global_zero else mute_tqdm
        for batch_idx, batch in enumerate(epoch_tqdm(train_dataloader, position=fabric.global_rank, pre=epoch_info,
                                                     desc=f"training", unit=f"x{train_dataloader.batch_size}")):
            args.prog.global_step += 1
            args.prog.global_epoch += args.prog.epoch_per_step
            batch: Dict[str, torch.Tensor] = pop_keys(batch, "example_ids")
            outputs: TokenClassifierOutput = model(**batch)
            labels: torch.Tensor = batch["labels"]
            preds: torch.Tensor = outputs.logits.argmax(dim=-1)
            acc: torch.Tensor = accuracy(preds, labels, ignore_index=0)
            metrics["epoch"] = round(args.prog.global_epoch, 4)
            metrics["trained_rate"] = round(args.prog.global_epoch, 4) / args.learning.num_epochs
            metrics["loss"] = outputs.loss.item()
            metrics["acc"] = acc.item()
            fabric.backward(outputs.loss)
            fabric.clip_gradients(model, optimizer, clip_val=0.25)
            optimizer.step()
            optimizer.zero_grad()
            model.eval()
            if batch_idx + 1 == len(train_dataloader) or (batch_idx + 1) % val_interval < 1:
                validate(fabric, args, model, valid_dataloader, valid_dataset, metrics=metrics, print_result=args.learning.tag_format_on_validate is not None)
                sorted_checkpoints = save_checkpoint(fabric, args, metrics, model, optimizer,
                                                     sorted_checkpoints, sorting_reverse, sorting_metric)
            fabric.log_dict(step=args.prog.global_step, metrics=metrics)
            model.train()
        scheduler.step()
        metrics["lr"] = optimizer.param_groups[0]['lr']
        if epoch + 1 < args.learning.num_epochs:
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
            for token_id in range(args.model.seq_len):
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
    metrics["val_F1c"] = klue_ner_char_macro_f1(preds=char_preds, labels=char_labels, label_list=valid_dataset.get_labels())
    metrics["val_F1e"] = klue_ner_entity_macro_f1(preds=char_preds, labels=char_labels, label_list=valid_dataset.get_labels())
    if print_result:
        terms = [m.group(1) for m in TERM_IN_NAME_FORMAT.finditer(args.learning.tag_format_on_validate)]
        terms = {term: metrics[term] for term in terms}
        fabric.print(' | ' + args.learning.tag_format_on_validate.format(**terms))


@app.command()
def train(args_file: Path | str):
    nlpbook.set_logger()
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args: TrainerArguments = TrainerArguments.from_json(args_file.read_text()).info_args()
    nlpbook.set_seed(args)

    with JobTimer(f"chrialab.nlpbook.ner train {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        if args.data.redownload:
            nlpbook.download_downstream_dataset(args)
            err_hr(c='-')

        corpus = NERCorpus(args)
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        train_dataset = NERDataset("train", corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.train_batch,
                                      collate_fn=corpus.encoded_examples_to_batch,
                                      drop_last=False)
        err_hr(c='-')

        val_dataset = NERDataset("valid", corpus=corpus, tokenizer=tokenizer)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    num_workers=args.hardware.cpu_workers,
                                    batch_size=args.hardware.infer_batch,
                                    collate_fn=corpus.encoded_examples_to_batch,
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

        with RuntimeChecking(args.configure_csv_logger()):
            torch.set_float32_matmul_precision('high')
            trainer: pl.Trainer = nlpbook.make_trainer(args)
            trainer.fit(NERTask(args,
                                model=model,
                                trainer=trainer,
                                epoch_steps=len(train_dataloader),
                                test_dataset=val_dataset),
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataset)


@app.command()
def test(args_file: Path | str):
    nlpbook.set_logger()
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args = TesterArguments.from_json(args_file.read_text()).info_args()

    with JobTimer(f"chrialab.nlpbook.ner test {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        checkpoint_path = args.env.output_home / args.model.name
        assert checkpoint_path.exists(), f"No checkpoint file: {checkpoint_path}"
        logger.info(f"Using finetuned checkpoint file at {checkpoint_path}")
        err_hr(c='-')

        nlpbook.download_downstream_dataset(args)
        err_hr(c='-')

        corpus = NERCorpus(args)
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        test_dataset = NERDataset("test", corpus=corpus, tokenizer=tokenizer)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.hardware.infer_batch,
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
    with RuntimeChecking(args.configure_csv_logger()):
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
    args: ServerArguments = ServerArguments.from_json(args_file.read_text()).info_args()

    with JobTimer(f"chrialab.nlpbook serve_ner {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        checkpoint_path = args.env.output_home / args.model.name
        assert checkpoint_path.exists(), f"No checkpoint file: {checkpoint_path}"
        checkpoint: dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        logger.info(f"Using finetuned checkpoint file at {checkpoint_path}")
        err_hr(c='-')

        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
        label_map_path: Path = args.env.output_home / "label_map.txt"
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
                max_length=args.model.seq_len,
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

        with RuntimeChecking(args.configure_csv_logger()):
            server: Flask = nlpbook.make_server(inference_fn,
                                                template_file="serve_ner.html",
                                                ngrok_home=args.env.working_dir)
            server.run()
