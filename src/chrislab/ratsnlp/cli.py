from pathlib import Path

import torch
from Korpora import Korpora
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import SequentialSampler
from typer import Typer

from chrisbase.io import JobTimer, out_hr
from ratsnlp import nlpbook
from ratsnlp.nlpbook.deploy import get_web_service_app
from ratsnlp.nlpbook.arguments import NLUTrainerArguments, NLUServerArguments
from ratsnlp.nlpbook.classification.corpus import NsmcCorpus, ClassificationDataset
from ratsnlp.nlpbook.classification.task import ClassificationTask
from ratsnlp.nlpbook.ner.corpus import NERCorpus, NERDataset
from ratsnlp.nlpbook.ner.task import NERTask
from transformers import BertConfig, BertForSequenceClassification, BertForTokenClassification
from transformers import BertTokenizer

app = Typer()


@app.command()
def train_ner(config: Path | str):
    config = Path(config)
    assert config.exists(), f"No config file: {config}"
    args = NLUTrainerArguments.from_json(config.read_text())
    args.print_dataframe()

    with JobTimer(f"chrialab.ratsnlp train_ner {config}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        nlpbook.set_seed(args)
        nlpbook.set_logger()
        out_hr(c='-')

        nlpbook.download_downstream_dataset(args)
        out_hr(c='-')

        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model_path,
            do_lower_case=False,
        )
        print(f"tokenizer={tokenizer}")
        print(f"tokenized={tokenizer.tokenize('안녕하세요. 반갑습니다.')}")
        out_hr(c='-')

        corpus = NERCorpus(args)
        train_dataset = NERDataset(
            args=args,
            corpus=corpus,
            tokenizer=tokenizer,
            mode="train",
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_dataset, replacement=False),
            collate_fn=nlpbook.data_collator,
            drop_last=False,
            num_workers=args.cpu_workers,
        )
        out_hr(c='-')

        val_dataset = NERDataset(
            args=args,
            corpus=corpus,
            tokenizer=tokenizer,
            mode="val",
        )
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
        model = BertForTokenClassification.from_pretrained(
            args.pretrained_model_path,
            config=pretrained_model_config,
        )
        out_hr(c='-')

        torch.set_float32_matmul_precision('high')
        nlpbook.get_trainer(args).fit(
            NERTask(model, args),
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )


@app.command()
def train_cls(config: Path | str):
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

        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model_path,
            do_lower_case=False,
        )
        print(f"tokenizer={tokenizer}")
        print(f"tokenized={tokenizer.tokenize('안녕하세요. 반갑습니다.')}")
        out_hr(c='-')

        corpus = NsmcCorpus()
        train_dataset = ClassificationDataset(
            args=args,
            corpus=corpus,
            tokenizer=tokenizer,
            mode="train",
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_dataset, replacement=False),
            collate_fn=nlpbook.data_collator,
            drop_last=False,
            num_workers=args.cpu_workers,
        )
        out_hr(c='-')

        val_dataset = ClassificationDataset(
            args=args,
            corpus=corpus,
            tokenizer=tokenizer,
            mode="test",
        )
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
def serve_ner(config: Path | str):
    config = Path(config)
    assert config.exists(), f"No config file: {config}"
    args = NLUServerArguments.from_json(config.read_text())
    args.print_dataframe()

    with JobTimer(f"chrialab.ratsnlp serve {config}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        downstream_model_path = args.downstream_model_home / args.downstream_model_file
        assert downstream_model_path.exists(), f"No downstream model file: {downstream_model_path}"
        downstream_model_ckpt = torch.load(downstream_model_path, map_location=torch.device("cpu"))
        pretrained_model_config = BertConfig.from_pretrained(
            args.pretrained_model_path,
            num_labels=downstream_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
        )
        model = BertForTokenClassification(pretrained_model_config)
        model.load_state_dict({k.replace("model.", ""): v for k, v in downstream_model_ckpt['state_dict'].items()})
        model.eval()

        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model_path,
            do_lower_case=False,
        )

        downstream_label_path: Path = args.downstream_model_home / "label_map.txt"
        assert downstream_label_path.exists(), f"No downstream label file: {downstream_label_path}"
        labels = downstream_label_path.read_text().splitlines(keepends=False)
        id_to_label = {idx: label for idx, label in enumerate(labels)}

        def inference_fn(sentence):
            from transformers.modeling_outputs import TokenClassifierOutput
            from torch import Tensor
            inputs = tokenizer(
                [sentence],
                max_length=args.max_seq_length,
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

        service = get_web_service_app(inference_fn, template_file="serve_ner.html", ngrok_home=args.env.working_path)
        service.run()


@app.command()
def serve_cls(config: Path | str):
    config = Path(config)
    assert config.exists(), f"No config file: {config}"
    args = NLUServerArguments.from_json(config.read_text())
    args.print_dataframe()

    with JobTimer(f"chrialab.ratsnlp serve {config}", mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
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

        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model_path,
            do_lower_case=False,
        )

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
