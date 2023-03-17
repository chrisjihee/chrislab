from pathlib import Path

import torch
from Korpora import Korpora
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import SequentialSampler
from typer import Typer

from chrisbase.io import JobTimer, out_hr, out_table
from chrisbase.util import to_dataframe
from ratsnlp import nlpbook
from ratsnlp.nlpbook.classification import ClassificationTask
from ratsnlp.nlpbook.classification import ClassificationTrainArguments
from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer

app = Typer()


@app.command()
def check(config: str, prefix: str = "", postfix: str = ""):
    print(f"config={config}, prefix={prefix}, postfix={postfix}")


@app.command()
def train(config: str, prefix: str = "", postfix: str = ""):
    config = Path(config)
    args = ClassificationTrainArguments.from_json(config.read_text())
    out_table(to_dataframe(args, columns=[args.__class__.__name__, "value"]))
    out_hr(c='-')

    with JobTimer(f"chrialab.ratsnlp train(config={config}, prefix={prefix}, postfix={postfix})",
                  mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        nlpbook.set_seed(args)
        nlpbook.set_logger()
        out_hr(c='-')

        Korpora.fetch(
            corpus_name=args.downstream_corpus_name,
            root_dir=args.downstream_corpus_root_dir,
        )
        out_hr(c='-')

        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model_name,
            do_lower_case=False,
        )
        print(f"tokenizer={tokenizer}")
        print(f"tokenized example={tokenizer.tokenize('안녕하세요. 반갑습니다.')}")
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
            args.pretrained_model_name,
            num_labels=corpus.num_labels,
        )
        model = BertForSequenceClassification.from_pretrained(
            args.pretrained_model_name,
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
def apply(config: str, prefix: str = "", postfix: str = ""):
    print(f"config={config}, prefix={prefix}, postfix={postfix}")
