from pathlib import Path

import pytorch_lightning as pl
import torch
from Korpora import Korpora
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import SequentialSampler
from typer import Typer

import nlpbook
from chrisbase.io import JobTimer, out_hr
from nlpbook.arguments import TrainerArguments, ServerArguments, CheckingRuntime
from nlpbook.cls.corpus import NsmcCorpus, ClassificationDataset
from nlpbook.cls.task import ClassificationTask
from nlpbook.deploy import get_web_service_app
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

app = Typer()


@app.command()
def train(args_file: Path | str):
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file file: {args_file}"
    args = TrainerArguments.from_json(args_file.read_text()).print_dataframe()

    with JobTimer(f"chrialab.nlpbook train_cls {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        nlpbook.set_seed(args)
        nlpbook.set_logger()
        out_hr(c='-')

        if not (args.model.data_home / args.model.data_name).exists() or not (args.model.data_home / args.model.data_name).is_dir():
            Korpora.fetch(
                corpus_name=args.model.data_name,
                root_dir=args.model.data_home,
            )
            out_hr(c='-')

        corpus = NsmcCorpus()
        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained_name, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.model.pretrained_name, do_lower_case=False)
        train_dataset = ClassificationDataset("train", args=args, corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.hardware.batch_size,
                                      num_workers=args.hardware.cpu_workers,
                                      sampler=RandomSampler(train_dataset, replacement=False),
                                      collate_fn=nlpbook.data_collator,
                                      drop_last=False)
        out_hr(c='-')

        val_dataset = ClassificationDataset("valid", args=args, corpus=corpus, tokenizer=tokenizer)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.hardware.batch_size,
                                    num_workers=args.hardware.cpu_workers,
                                    sampler=SequentialSampler(val_dataset),
                                    collate_fn=nlpbook.data_collator,
                                    drop_last=False)
        out_hr(c='-')

        pretrained_model_config = BertConfig.from_pretrained(
            args.model.pretrained_name,
            num_labels=corpus.num_labels,
        )
        model = BertForSequenceClassification.from_pretrained(
            args.model.pretrained_name,
            config=pretrained_model_config,
        )
        out_hr(c='-')

        torch.set_float32_matmul_precision('high')
        trainer: pl.Trainer = nlpbook.get_trainer(args)
        pl_module: pl.LightningModule = ClassificationTask(model, args, trainer)
        with CheckingRuntime(args, trainer.logger.log_dir):
            trainer.fit(pl_module,
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)


@app.command()
def serve(args_file: Path | str):
    args_file = Path(args_file)
    assert args_file.exists(), f"No args_file file: {args_file}"
    args = ServerArguments.from_json(args_file.read_text())
    args.print_dataframe()

    with JobTimer(f"chrialab.nlpbook serve_cls {args_file}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        downstream_model_path = args.model.finetuned_home / args.model.finetuned_name
        assert downstream_model_path.exists(), f"No downstream model file: {downstream_model_path}"
        downstream_model_ckpt = torch.load(downstream_model_path, map_location=torch.device("cpu"))
        pretrained_model_config = BertConfig.from_pretrained(
            args.model.pretrained_name,
            num_labels=downstream_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
        )
        model = BertForSequenceClassification(pretrained_model_config)
        model.load_state_dict({k.replace("model.", ""): v for k, v in downstream_model_ckpt['state_dict'].items()})
        model.eval()

        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model.pretrained_name, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.model.pretrained_name, do_lower_case=False)

        def inference_fn(sentence):
            inputs = tokenizer(
                [sentence],
                max_length=args.model.max_seq_length,
                padding="max_length",
                truncation=True,
            )
            with torch.no_grad():
                outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
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

        service = get_web_service_app(inference_fn, template_file="serve_cls.html", ngrok_home=args.env.working_path)
        service.run()
