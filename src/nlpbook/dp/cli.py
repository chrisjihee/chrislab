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
from nlpbook.metrics import accuracy
from nlpbook.ner.corpus import NERCorpus, NERDataset, encoded_examples_to_batch, NEREncodedExample
from nlpbook.ner.task import NERTask
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoConfig, AutoModelForTokenClassification, BertForTokenClassification, CharSpan
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
