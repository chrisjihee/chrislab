import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, ClassVar, Dict

import torch
from dataclasses_json import DataClassJsonMixin
from torch.utils.data.dataset import Dataset

from chrisbase.io import make_parent_dir, files, merge_dicts, out_hr, sys_stderr
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


@dataclass
class DPEncodedExample:
    idx: int
    raw: DPRawExample
    encoded: BatchEncoding
    ids: List[int]
    mask: List[int]
    bpe_head_mask: List[int]
    bpe_tail_mask: List[int]
    head_ids: List[int]
    dep_ids: List[int]
    pos_ids: List[int]


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


def _convert_to_encoded_examples(
        raw_examples: List[DPRawExample],
        tokenizer: PreTrainedTokenizerFast,
        args: TesterArguments,
        pos_label_list: List[str],
        dep_label_list: List[str],
) -> List[DPEncodedExample]:
    pos_label_to_id = {label: i for i, label in enumerate(pos_label_list)}
    dep_label_to_id = {label: i for i, label in enumerate(dep_label_list)}
    id_to_pos_label = {i: label for i, label in enumerate(pos_label_list)}
    id_to_dep_label = {i: label for i, label in enumerate(dep_label_list)}
    if args.env.off_debugging:
        print(f"pos_label_to_id = {pos_label_to_id}")
        print(f"dep_label_to_id = {dep_label_to_id}")
        print(f"id_to_pos_label = {id_to_pos_label}")
        print(f"id_to_dep_label = {id_to_dep_label}")

    SENT_ID = 0
    token_list: List[str] = []
    pos_list: List[str] = []
    head_list: List[int] = []
    dep_list: List[str] = []

    encoded_examples: List[DPEncodedExample] = []
    prev_raw_example: Optional[DPRawExample] = None
    prev_SENT_ID: int = -1
    for raw_example in raw_examples:
        raw_example: DPRawExample = raw_example
        if SENT_ID != raw_example.sent_id:
            SENT_ID = raw_example.sent_id
            encoded: BatchEncoding = tokenizer.encode_plus(" ".join(token_list),
                                                           max_length=args.model.max_seq_length,
                                                           truncation=TruncationStrategy.LONGEST_FIRST,
                                                           padding=PaddingStrategy.MAX_LENGTH)
            if args.env.off_debugging:
                out_hr()
                print(f"encoded.tokens()        = {encoded.tokens()}")
                for key in encoded.keys():
                    print(f"encoded[{key:14s}] = {encoded[key]}")
            ids, mask = encoded.input_ids, encoded.attention_mask

            # TODO: 추후 encoded.word_to_tokens() 함수를 활용한 코드로 변경!
            bpe_head_mask = [0]
            bpe_tail_mask = [0]
            head_ids = [-1]
            dep_ids = [-1]
            pos_ids = [-1]  # --> CLS token
            for token, head, dep, pos in zip(token_list, head_list, dep_list, pos_list):
                bpe_len = len(tokenizer.tokenize(token))
                head_token_mask = [1] + [0] * (bpe_len - 1)
                tail_token_mask = [0] * (bpe_len - 1) + [1]
                head_mask = [head] + [-1] * (bpe_len - 1)
                dep_mask = [dep_label_to_id[dep]] + [-1] * (bpe_len - 1)
                pos_mask = [pos_label_to_id[pos]] + [-1] * (bpe_len - 1)
                bpe_head_mask.extend(head_token_mask)
                bpe_tail_mask.extend(tail_token_mask)
                head_ids.extend(head_mask)
                dep_ids.extend(dep_mask)
                pos_ids.extend(pos_mask)
            bpe_head_mask.append(0)
            bpe_tail_mask.append(0)
            head_ids.append(-1)
            dep_ids.append(-1)
            pos_ids.append(-1)  # --> SEP token
            if len(bpe_head_mask) > args.model.max_seq_length:
                bpe_head_mask = bpe_head_mask[:args.model.max_seq_length]
                bpe_tail_mask = bpe_tail_mask[:args.model.max_seq_length]
                head_ids = head_ids[:args.model.max_seq_length]
                dep_ids = dep_ids[:args.model.max_seq_length]
                pos_ids = pos_ids[:args.model.max_seq_length]
            else:
                bpe_head_mask.extend([0] * (args.model.max_seq_length - len(bpe_head_mask)))
                bpe_tail_mask.extend([0] * (args.model.max_seq_length - len(bpe_tail_mask)))
                head_ids.extend([-1] * (args.model.max_seq_length - len(head_ids)))
                dep_ids.extend([-1] * (args.model.max_seq_length - len(dep_ids)))
                pos_ids.extend([-1] * (args.model.max_seq_length - len(pos_ids)))

            encoded_example = DPEncodedExample(
                idx=prev_SENT_ID,
                raw=prev_raw_example,
                encoded=encoded,
                ids=ids,
                mask=mask,
                bpe_head_mask=bpe_head_mask,
                bpe_tail_mask=bpe_tail_mask,
                head_ids=head_ids,
                dep_ids=dep_ids,
                pos_ids=pos_ids,
            )
            encoded_examples.append(encoded_example)
            if args.env.off_debugging:
                out_hr()
                print(f"bpe_head_mask           = {bpe_head_mask}")
                print(f"bpe_tail_mask           = {bpe_tail_mask}")
                print(f"head_ids                = {head_ids}")
                print(f"dep_ids                 = {dep_ids}")
                print(f"pos_ids                 = {pos_ids}")
                print()

            token_list = []
            pos_list = []
            head_list = []
            dep_list = []

        token_list.append(raw_example.token)
        pos_list.append(raw_example.pos.split("+")[-1])  # 맨 뒤 pos정보만 사용
        head_list.append(int(raw_example.head))
        dep_list.append(raw_example.dep)
        prev_raw_example = raw_example
        prev_SENT_ID = SENT_ID

    encoded: BatchEncoding = tokenizer.encode_plus(" ".join(token_list),
                                                   max_length=args.model.max_seq_length,
                                                   truncation=TruncationStrategy.LONGEST_FIRST,
                                                   padding=PaddingStrategy.MAX_LENGTH)
    if args.env.off_debugging:
        out_hr()
        print(f"encoded.tokens()        = {encoded.tokens()}")
        for key in encoded.keys():
            print(f"encoded[{key:14s}] = {encoded[key]}")
    ids, mask = encoded.input_ids, encoded.attention_mask

    # TODO: 추후 encoded.word_to_tokens() 함수를 활용한 코드로 변경!
    bpe_head_mask = [0]
    bpe_tail_mask = [0]
    head_ids = [-1]
    dep_ids = [-1]
    pos_ids = [-1]  # --> CLS token
    for token, head, dep, pos in zip(token_list, head_list, dep_list, pos_list):
        bpe_len = len(tokenizer.tokenize(token))
        head_token_mask = [1] + [0] * (bpe_len - 1)
        tail_token_mask = [0] * (bpe_len - 1) + [1]
        head_mask = [head] + [-1] * (bpe_len - 1)
        dep_mask = [dep_label_to_id[dep]] + [-1] * (bpe_len - 1)
        pos_mask = [pos_label_to_id[pos]] + [-1] * (bpe_len - 1)
        bpe_head_mask.extend(head_token_mask)
        bpe_tail_mask.extend(tail_token_mask)
        head_ids.extend(head_mask)
        dep_ids.extend(dep_mask)
        pos_ids.extend(pos_mask)
    bpe_head_mask.append(0)
    bpe_tail_mask.append(0)
    head_ids.append(-1)
    dep_ids.append(-1)
    pos_ids.append(-1)  # --> SEP token
    bpe_head_mask.extend([0] * (args.model.max_seq_length - len(bpe_head_mask)))
    bpe_tail_mask.extend([0] * (args.model.max_seq_length - len(bpe_tail_mask)))
    head_ids.extend([-1] * (args.model.max_seq_length - len(head_ids)))
    dep_ids.extend([-1] * (args.model.max_seq_length - len(dep_ids)))
    pos_ids.extend([-1] * (args.model.max_seq_length - len(pos_ids)))

    encoded_example = DPEncodedExample(
        idx=prev_SENT_ID,
        raw=prev_raw_example,
        encoded=encoded,
        ids=ids,
        mask=mask,
        bpe_head_mask=bpe_head_mask,
        bpe_tail_mask=bpe_tail_mask,
        head_ids=head_ids,
        dep_ids=dep_ids,
        pos_ids=pos_ids,
    )
    encoded_examples.append(encoded_example)
    if args.env.off_debugging:
        out_hr()
        print(f"bpe_head_mask           = {bpe_head_mask}")
        print(f"bpe_tail_mask           = {bpe_tail_mask}")
        print(f"head_ids                = {head_ids}")
        print(f"dep_ids                 = {dep_ids}")
        print(f"pos_ids                 = {pos_ids}")
        print()

    if args.env.off_debugging:
        out_hr()
    for encoded_example in encoded_examples[:args.data.show_examples]:
        logger.info("  === [Example %s] ===" % encoded_example.idx)
        logger.info("  = sentence      : %s" % encoded_example.raw.text)
        logger.info("  = tokens        : %s" % " ".join(encoded_example.encoded.tokens()))
        logger.info("  = bpe_head_mask : %s" % " ".join(str(x) for x in encoded_example.bpe_head_mask))
        logger.info("  = bpe_tail_mask : %s" % " ".join(str(x) for x in encoded_example.bpe_tail_mask))
        logger.info("  = head_ids      : %s" % " ".join(str(x) for x in encoded_example.head_ids))
        logger.info("  = dep_labels    : %s" % " ".join(id_to_dep_label[x] if x in id_to_dep_label else str(x) for x in encoded_example.dep_ids))
        logger.info("  = pos_labels    : %s" % " ".join(id_to_pos_label[x] if x in id_to_pos_label else str(x) for x in encoded_example.pos_ids))
        logger.info("  === ")

    logger.info(f"Converted {len(raw_examples)} raw examples to {len(encoded_examples)} encoded examples")
    return encoded_examples


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
        self._dep_label_to_id: Dict[str, int] = {label: i for i, label in enumerate(self.dep_labels)}
        self._pos_label_to_id: Dict[str, int] = {label: i for i, label in enumerate(self.pos_labels)}
        self._id_to_dep_label: Dict[int, str] = {i: label for i, label in enumerate(self.dep_labels)}
        self._id_to_pos_label: Dict[int, str] = {i: label for i, label in enumerate(self.pos_labels)}
        self.features: List[DPEncodedExample] = \
            _convert_to_encoded_examples(examples, tokenizer, args,
                                         pos_label_list=self.pos_labels,
                                         dep_label_list=self.dep_labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i) -> DPEncodedExample:
        return self.features[i]

    def get_dep_labels(self) -> List[str]:
        return self.dep_labels

    def get_pos_labels(self) -> List[str]:
        return self.pos_labels

    def dep_label_to_id(self, label: str) -> int:
        return self._dep_label_to_id[label]

    def pos_label_to_id(self, label: str) -> int:
        return self._pos_label_to_id[label]

    def id_to_dep_label(self, label_id: int) -> str:
        return self._id_to_dep_label[label_id]

    def id_to_pos_label(self, label_id: int) -> str:
        return self._id_to_pos_label[label_id]
