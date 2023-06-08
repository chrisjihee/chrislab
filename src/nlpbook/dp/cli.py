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
