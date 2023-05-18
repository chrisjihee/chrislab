import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from flask import Flask
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import SequentialSampler
from typer import Typer

import nlpbook
from chrisbase.io import JobTimer, err_hr
from nlpbook.arguments import TrainerArguments, ServerArguments, TesterArguments, RuntimeChecking
from nlpbook.ner.corpus import NERCorpus, NERDataset
from nlpbook.ner.task import NERTask
from transformers import BertConfig, BertForTokenClassification, BertTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

app = Typer()
logger = logging.getLogger("nlpbook")


@app.command()
def train(args_file: Path | str):
    nlpbook.set_logger()
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args: TrainerArguments = TrainerArguments.from_json(args_file.read_text()).show()
    nlpbook.set_seed(args)

    with JobTimer(f"chrialab.nlpbook.ner train {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        nlpbook.download_downstream_dataset(args)
        err_hr(c='-')

        corpus = NERCorpus(args)
        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False)
        train_dataset = NERDataset("train", args=args, corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.hardware.batch_size,
                                      num_workers=args.hardware.cpu_workers,
                                      sampler=RandomSampler(train_dataset, replacement=False),
                                      collate_fn=nlpbook.data_collator,
                                      drop_last=False)
        err_hr(c='-')

        val_dataset = NERDataset("valid", args=args, corpus=corpus, tokenizer=tokenizer)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.hardware.batch_size,
                                    num_workers=args.hardware.cpu_workers,
                                    sampler=SequentialSampler(val_dataset),
                                    collate_fn=nlpbook.data_collator,
                                    drop_last=False)
        err_hr(c='-')

        pretrained_model_config = BertConfig.from_pretrained(
            args.model.pretrained,
            num_labels=corpus.num_labels
        )
        model = BertForTokenClassification.from_pretrained(
            args.model.pretrained,
            config=pretrained_model_config
        )
        err_hr(c='-')

        with RuntimeChecking(nlpbook.setup_csv_out(args)):
            torch.set_float32_matmul_precision('high')
            trainer: pl.Trainer = nlpbook.make_trainer(args)
            trainer.fit(NERTask(model, args, trainer),
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
        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False)
        test_dataset = NERDataset("test", args=args, corpus=corpus, tokenizer=tokenizer)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.hardware.batch_size,
                                     num_workers=args.hardware.cpu_workers,
                                     sampler=SequentialSampler(test_dataset),
                                     collate_fn=nlpbook.data_collator,
                                     drop_last=False)
        err_hr(c='-')

        pretrained_model_config = BertConfig.from_pretrained(
            args.model.pretrained,
            num_labels=corpus.num_labels
        )
        model = BertForTokenClassification.from_pretrained(
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

        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False)
        label_map_path: Path = args.output.dir_path / "label_map.txt"
        assert label_map_path.exists(), f"No downstream label file: {label_map_path}"
        labels = label_map_path.read_text().splitlines(keepends=False)
        id_to_label = {idx: label for idx, label in enumerate(labels)}

        pretrained_model_config = BertConfig.from_pretrained(
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
