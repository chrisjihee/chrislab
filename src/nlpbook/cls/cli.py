import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from Korpora import Korpora
from flask import Flask
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import SequentialSampler
from typer import Typer

import nlpbook
from chrisbase.io import JobTimer, err_hr
from nlpbook.arguments import TrainerArguments, ServerArguments, TesterArguments, RuntimeChecking
from nlpbook.cls.corpus import NsmcCorpus, ClassificationDataset
from nlpbook.cls.task import ClassificationTask
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

app = Typer()
logger = logging.getLogger("nlpbook")


@app.command()
def train(args_file: Path | str):
    nlpbook.set_logger()
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args: TrainerArguments = TrainerArguments.from_json(args_file.read_text()).show()
    nlpbook.set_seed(args)

    with JobTimer(f"chrialab.nlpbook.cls train {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        if not (args.data.home / args.data.name).exists() or not (args.data.home / args.data.name).is_dir():
            Korpora.fetch(
                corpus_name=args.data.name,
                root_dir=args.data.home,
            )
        err_hr(c='-')

        corpus = NsmcCorpus()
        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False)
        train_dataset = ClassificationDataset("train", args=args, corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.hardware.batch_size,
                                      num_workers=args.hardware.cpu_workers,
                                      sampler=RandomSampler(train_dataset, replacement=False),
                                      collate_fn=nlpbook.data_collator,
                                      drop_last=False)
        err_hr(c='-')

        val_dataset = ClassificationDataset("valid", args=args, corpus=corpus, tokenizer=tokenizer)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.hardware.batch_size,
                                    num_workers=args.hardware.cpu_workers,
                                    sampler=SequentialSampler(val_dataset),
                                    collate_fn=nlpbook.data_collator,
                                    drop_last=False)
        err_hr(c='-')

        pretrained_model_config = BertConfig.from_pretrained(
            args.model.pretrained,
            num_labels=corpus.num_labels,
        )
        model = BertForSequenceClassification.from_pretrained(
            args.model.pretrained,
            config=pretrained_model_config,
        )
        err_hr(c='-')

        with RuntimeChecking(nlpbook.setup_csv_out(args)):
            torch.set_float32_matmul_precision('high')
            trainer: pl.Trainer = nlpbook.make_trainer(args)
            trainer.fit(ClassificationTask(model, args, trainer),
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)


@app.command()
def test(args_file: Path | str):
    nlpbook.set_logger()
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file: {args_file}"
    args: TesterArguments = TesterArguments.from_json(args_file.read_text()).show()

    with JobTimer(f"chrialab.nlpbook.cls test {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        checkpoint_path = args.output.dir_path / args.model.finetuning_name
        assert checkpoint_path.exists(), f"No checkpoint file: {checkpoint_path}"
        logger.info(f"Using finetuned checkpoint file at {checkpoint_path}")
        err_hr(c='-')

        if not (args.data.home / args.data.name).exists() or not (args.data.home / args.data.name).is_dir():
            Korpora.fetch(
                corpus_name=args.data.name,
                root_dir=args.data.home,
            )
        err_hr(c='-')

        corpus = NsmcCorpus()
        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False)
        test_dataset = ClassificationDataset("test", args=args, corpus=corpus, tokenizer=tokenizer)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.hardware.batch_size,
                                     num_workers=args.hardware.cpu_workers,
                                     sampler=SequentialSampler(test_dataset),
                                     collate_fn=nlpbook.data_collator,
                                     drop_last=False)
        err_hr(c='-')

        pretrained_model_config = BertConfig.from_pretrained(
            args.model.pretrained,
            num_labels=corpus.num_labels,
        )
        model = BertForSequenceClassification.from_pretrained(
            args.model.pretrained,
            config=pretrained_model_config,
        )
        err_hr(c='-')

        with RuntimeChecking(nlpbook.setup_csv_out(args)):
            torch.set_float32_matmul_precision('high')
            tester: pl.Trainer = nlpbook.make_tester(args)
            tester.test(ClassificationTask(model, args, tester),
                        dataloaders=test_dataloader,
                        ckpt_path=checkpoint_path)


@app.command()
def serve(args_file: Path | str):
    nlpbook.set_logger()
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file file: {args_file}"
    args: ServerArguments = ServerArguments.from_json(args_file.read_text()).show()

    with JobTimer(f"chrialab.nlpbook serve_cls {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        checkpoint_path = args.output.dir_path / args.model.finetuning_name
        assert checkpoint_path.exists(), f"No downstream model file: {checkpoint_path}"
        checkpoint: dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        logger.info(f"Using finetuned model file at {checkpoint_path}")
        err_hr(c='-')

        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.model.pretrained, do_lower_case=False)

        pretrained_model_config = BertConfig.from_pretrained(
            args.model.pretrained,
            num_labels=checkpoint['state_dict']['model.classifier.bias'].shape.numel(),
        )
        model = BertForSequenceClassification(pretrained_model_config)
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
                outputs: SequenceClassifierOutput = model(**{k: torch.tensor(v) for k, v in inputs.items()})
                prob = outputs.logits.softmax(dim=1)
                positive_prob = round(prob[0][1].item(), 4)
                negative_prob = round(prob[0][0].item(), 4)
                pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
            return {
                'sentence': sentence,
                'prediction': pred,
                'positive_data': f"긍정 {positive_prob:.4f}",
                'negative_data': f"부정 {negative_prob:.4f}",
                'positive_width': f"{positive_prob * 100}%",
                'negative_width': f"{negative_prob * 100}%",
            }

        with RuntimeChecking(nlpbook.setup_csv_out(args)):
            server: Flask = nlpbook.make_server(inference_fn,
                                                template_file="serve_cls.html",
                                                ngrok_home=args.env.working_path)
            server.run()
