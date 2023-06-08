import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, ClassVar, Dict

import torch
from dataclasses_json import DataClassJsonMixin
from torch.utils.data.dataset import Dataset

from chrisbase.io import make_parent_dir, files, merge_dicts, out_hr
from nlpbook.arguments import TesterArguments
from transformers import PreTrainedTokenizerFast, BatchEncoding, CharSpan
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

logger = logging.getLogger("chrislab")


@dataclass
class DPRawExample(DataClassJsonMixin):
    origin: str = field(default_factory=str)


class DPCorpus:
    def __init__(self, args: TesterArguments):
        self.args = args

    def read_raw_examples(self, data_path: Path) -> List[DPRawExample]:
        examples = []
        with data_path.open(encoding="utf-8") as inp:
            pass
        logger.info(f"Loaded {len(examples)} examples from {data_path}")
        return examples


class DPDataset(Dataset):
    def __init__(self, split: str, args: TesterArguments, tokenizer: PreTrainedTokenizerFast, corpus: DPCorpus):
        assert corpus, "corpus is not valid"
        assert args.data.home, f"No data_home: {args.data.home}"
        assert args.data.name, f"No data_name: {args.data.name}"
        self.corpus: DPCorpus = corpus
        data_file_dict: dict = args.data.files.to_dict()
        assert split in data_file_dict, f"No '{split}' split in data_file: should be one of {list(data_file_dict.keys())}"
        assert data_file_dict[split], f"No data_file for '{split}' split: {args.data.files}"
        text_data_path: Path = Path(args.data.home) / args.data.name / data_file_dict[split]
        assert text_data_path.exists() and text_data_path.is_file(), f"No data_text_path: {text_data_path}"
        logger.info(f"Creating features from dataset file at {text_data_path}")
        examples: List[DPRawExample] = self.corpus.read_raw_examples(text_data_path)
