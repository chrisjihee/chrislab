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
    guid: str = field()
    text: str = field()
    sent_id: int = field()
    token_id: int = field()
    token: str = field()
    pos: str = field()
    head: str = field()
    dep: str = field()


class DPCorpus:
    def __init__(self, args: TesterArguments):
        self.args = args
        self.dep_labels = [
            "NP",
            "NP_AJT",
            "VP",
            "NP_SBJ",
            "VP_MOD",
            "NP_OBJ",
            "AP",
            "NP_CNJ",
            "NP_MOD",
            "VNP",
            "DP",
            "VP_AJT",
            "VNP_MOD",
            "NP_CMP",
            "VP_SBJ",
            "VP_CMP",
            "VP_OBJ",
            "VNP_CMP",
            "AP_MOD",
            "X_AJT",
            "VP_CNJ",
            "VNP_AJT",
            "IP",
            "X",
            "X_SBJ",
            "VNP_OBJ",
            "VNP_SBJ",
            "X_OBJ",
            "AP_AJT",
            "L",
            "X_MOD",
            "X_CNJ",
            "VNP_CNJ",
            "X_CMP",
            "AP_CMP",
            "AP_SBJ",
            "R",
            "NP_SVJ",
        ]
        self.pos_labels = [
            "NNG",
            "NNP",
            "NNB",
            "NP",
            "NR",
            "VV",
            "VA",
            "VX",
            "VCP",
            "VCN",
            "MMA",
            "MMD",
            "MMN",
            "MAG",
            "MAJ",
            "JC",
            "IC",
            "JKS",
            "JKC",
            "JKG",
            "JKO",
            "JKB",
            "JKV",
            "JKQ",
            "JX",
            "EP",
            "EF",
            "EC",
            "ETN",
            "ETM",
            "XPN",
            "XSN",
            "XSV",
            "XSA",
            "XR",
            "SF",
            "SP",
            "SS",
            "SE",
            "SO",
            "SL",
            "SH",
            "SW",
            "SN",
            "NA",
        ]

    def read_raw_examples(self, data_path: Path) -> List[DPRawExample]:
        sent_id = -1
        examples = []
        with data_path.open(encoding="utf-8") as inp:
            for line in inp:
                line = line.strip()
                if line == "" or line == "\n" or line == "\t":
                    continue
                if line.startswith("#"):
                    parsed = line.strip().split("\t")
                    if len(parsed) != 2:  # metadata line about dataset
                        continue
                    else:
                        sent_id += 1
                        text = parsed[1].strip()
                        guid = parsed[0].replace("##", "").strip()
                else:
                    token_list = [token.replace("\n", "") for token in line.split("\t")] + ["-", "-"]
                    examples.append(
                        DPRawExample(
                            guid=guid,
                            text=text,
                            sent_id=sent_id,
                            token_id=int(token_list[0]),
                            token=token_list[1],
                            pos=token_list[3],
                            head=token_list[4],
                            dep=token_list[5],
                        )
                    )
        logger.info(f"Loaded {len(examples)} examples from {data_path}")
        return examples

    def get_dep_labels(self) -> List[str]:
        return self.dep_labels

    def get_pos_labels(self) -> List[str]:
        return self.pos_labels


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
        self.dep_labels: List[str] = self.corpus.get_dep_labels()
        self.pos_labels: List[str] = self.corpus.get_pos_labels()
