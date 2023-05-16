from pathlib import Path

import torch
from Korpora import Korpora
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import SequentialSampler
from typer import Typer

import nlpbook
from chrisbase.io import JobTimer, out_hr
from nlpbook.arguments import NLUTrainerArguments, NLUServerArguments
from nlpbook.cls.corpus import NsmcCorpus, ClassificationDataset
from nlpbook.cls.task import ClassificationTask
from nlpbook.deploy import get_web_service_app
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast, AutoTokenizer, BertTokenizer

app = Typer()


@app.command()
def train(config: Path | str):
    config = Path(config)
    assert config.exists(), f"No config file: {config}"
    args = NLUTrainerArguments.from_json(config.read_text())
    args.print_dataframe()

    with JobTimer(f"chrialab.ratsnlp train_cls {config}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        nlpbook.set_seed(args)
        nlpbook.set_logger()
        out_hr(c='-')

        Korpora.fetch(
            corpus_name=args.downstream_data_name,
            root_dir=args.downstream_data_home,
        )
        out_hr(c='-')

        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=False)
        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"
        print(f"tokenizer={tokenizer}")
        print(f"tokenized={tokenizer.tokenize('안녕하세요. 반갑습니다.')}")
        out_hr(c='-')

        corpus = NsmcCorpus()
        train_dataset = ClassificationDataset(args=args, corpus=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_dataset, replacement=False),
            collate_fn=nlpbook.data_collator,
            drop_last=False,
            num_workers=args.cpu_workers,
        )
        out_hr(c='-')

        val_dataset = ClassificationDataset(args=args, corpus=corpus, tokenizer=tokenizer)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(val_dataset),
            collate_fn=nlpbook.data_collator,
            drop_last=False,
            num_workers=args.cpu_workers,
        )
        out_hr(c='-')

        pretrained_model_config = BertConfig.from_pretrained(
            args.pretrained_model_path,
            num_labels=corpus.num_labels,
        )
        model = BertForSequenceClassification.from_pretrained(
            args.pretrained_model_path,
            config=pretrained_model_config,
        )
        out_hr(c='-')

        torch.set_float32_matmul_precision('high')
        nlpbook.get_trainer(args).fit(
            ClassificationTask(model, args),
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )


@app.command()
def serve(config: Path | str):
    config = Path(config)
    assert config.exists(), f"No config file: {config}"
    args = NLUServerArguments.from_json(config.read_text())
    args.print_dataframe()

    with JobTimer(f"chrialab.ratsnlp serve_cls {config}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        downstream_model_path = args.downstream_model_home / args.downstream_model_file
        assert downstream_model_path.exists(), f"No downstream model file: {downstream_model_path}"
        downstream_model_ckpt = torch.load(downstream_model_path, map_location=torch.device("cpu"))
        pretrained_model_config = BertConfig.from_pretrained(
            args.pretrained_model_path,
            num_labels=downstream_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
        )
        model = BertForSequenceClassification(pretrained_model_config)
        model.load_state_dict({k.replace("model.", ""): v for k, v in downstream_model_ckpt['state_dict'].items()})
        model.eval()

        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=False)

        # tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=False, use_fast=True)
        # assert isinstance(tokenizer, PreTrainedTokenizerFast), f"tokenizer is not PreTrainedTokenizerFast: {type(tokenizer)}"

        def inference_fn(sentence):
            inputs = tokenizer(
                [sentence],
                max_length=args.max_seq_length,
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
