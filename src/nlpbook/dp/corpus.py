import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import torch
from dataclasses_json import DataClassJsonMixin
from torch.utils.data.dataset import Dataset

from chrisbase.io import hr
from nlpbook.arguments import TesterArguments, TrainerArguments
from transformers import PreTrainedTokenizerFast, BatchEncoding
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

logger = logging.getLogger(__name__)


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
    bpe_head_mask: List[int]
    bpe_tail_mask: List[int]
    head_ids: List[int]
    dep_ids: List[int]
    pos_ids: List[int]


class DPCorpus:
    def __init__(self, args: TesterArguments | TrainerArguments):
        self.args = args

    dep_labels = [
        "NP", "NP_AJT", "VP", "NP_SBJ", "VP_MOD", "NP_OBJ", "AP", "NP_CNJ", "NP_MOD", "VNP",
        "DP", "VP_AJT", "VNP_MOD", "NP_CMP", "VP_SBJ", "VP_CMP", "VP_OBJ", "VNP_CMP", "AP_MOD", "X_AJT",
        "VP_CNJ", "VNP_AJT", "IP", "X", "X_SBJ", "VNP_OBJ", "VNP_SBJ", "X_OBJ", "AP_AJT", "L",
        "X_MOD", "X_CNJ", "VNP_CNJ", "X_CMP", "AP_CMP", "AP_SBJ", "R", "NP_SVJ",
    ]
    pos_labels = [
        "NNG", "NNP", "NNB", "NP", "NR", "VV", "VA", "VX", "VCP", "VCN",
        "MMA", "MMD", "MMN", "MAG", "MAJ", "JC", "IC", "JKS", "JKC", "JKG",
        "JKO", "JKB", "JKV", "JKQ", "JX", "EP", "EF", "EC", "ETN", "ETM",
        "XPN", "XSN", "XSV", "XSA", "XR", "SF", "SP", "SS", "SE", "SO",
        "SL", "SH", "SW", "SN", "NA",
    ]

    @property
    def num_labels(self) -> int:
        return len(self.get_dep_labels())

    @classmethod
    def get_dep_labels(cls) -> List[str]:
        return cls.dep_labels

    @classmethod
    def get_pos_labels(cls) -> List[str]:
        return cls.pos_labels

    def read_raw_examples(self, split: str) -> List[DPRawExample]:
        assert self.args.data.home, f"No data_home: {self.args.data.home}"
        assert self.args.data.name, f"No data_name: {self.args.data.name}"
        data_file_dict: dict = self.args.data.files.to_dict()
        assert split in data_file_dict, f"No '{split}' split in data_file: should be one of {list(data_file_dict.keys())}"
        assert data_file_dict[split], f"No data_file for '{split}' split: {self.args.data.files}"
        data_path: Path = Path(self.args.data.home) / self.args.data.name / data_file_dict[split]
        assert data_path.exists() and data_path.is_file(), f"No data_text_path: {data_path}"
        logger.info(f"Creating features from {data_path}")

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

    def raw_examples_to_encoded_examples(
            self,
            raw_examples: List[DPRawExample],
            tokenizer: PreTrainedTokenizerFast,
            pos_label_list: List[str],
            dep_label_list: List[str],
    ) -> List[DPEncodedExample]:
        pos_label_to_id = {label: i for i, label in enumerate(pos_label_list)}
        dep_label_to_id = {label: i for i, label in enumerate(dep_label_list)}
        id_to_pos_label = {i: label for i, label in enumerate(pos_label_list)}
        id_to_dep_label = {i: label for i, label in enumerate(dep_label_list)}
        logger.debug(f"pos_label_to_id = {pos_label_to_id}")
        logger.debug(f"dep_label_to_id = {dep_label_to_id}")
        logger.debug(f"id_to_pos_label = {id_to_pos_label}")
        logger.debug(f"id_to_dep_label = {id_to_dep_label}")

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
                                                               max_length=self.args.model.seq_len,
                                                               truncation=TruncationStrategy.LONGEST_FIRST,
                                                               padding=PaddingStrategy.MAX_LENGTH)
                logger.debug(hr())
                logger.debug(f"encoded.tokens()        = {encoded.tokens()}")
                for key in encoded.keys():
                    logger.debug(f"encoded[{key:14s}] = {encoded[key]}")

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
                if len(bpe_head_mask) > self.args.model.seq_len:
                    bpe_head_mask = bpe_head_mask[:self.args.model.seq_len]
                    bpe_tail_mask = bpe_tail_mask[:self.args.model.seq_len]
                    head_ids = head_ids[:self.args.model.seq_len]
                    dep_ids = dep_ids[:self.args.model.seq_len]
                    pos_ids = pos_ids[:self.args.model.seq_len]
                else:
                    bpe_head_mask.extend([0] * (self.args.model.seq_len - len(bpe_head_mask)))
                    bpe_tail_mask.extend([0] * (self.args.model.seq_len - len(bpe_tail_mask)))
                    head_ids.extend([-1] * (self.args.model.seq_len - len(head_ids)))
                    dep_ids.extend([-1] * (self.args.model.seq_len - len(dep_ids)))
                    pos_ids.extend([-1] * (self.args.model.seq_len - len(pos_ids)))

                encoded_example = DPEncodedExample(
                    idx=prev_SENT_ID,
                    raw=prev_raw_example,
                    encoded=encoded,
                    bpe_head_mask=bpe_head_mask,
                    bpe_tail_mask=bpe_tail_mask,
                    head_ids=head_ids,
                    dep_ids=dep_ids,
                    pos_ids=pos_ids,
                )
                encoded_examples.append(encoded_example)
                logger.debug(hr())
                logger.debug(f"bpe_head_mask           = {bpe_head_mask}")
                logger.debug(f"bpe_tail_mask           = {bpe_tail_mask}")
                logger.debug(f"head_ids                = {head_ids}")
                logger.debug(f"dep_ids                 = {dep_ids}")
                logger.debug(f"pos_ids                 = {pos_ids}")
                logger.debug("")

                token_list = []
                head_list = []
                dep_list = []
                pos_list = []

            token_list.append(raw_example.token)
            pos_list.append(raw_example.pos.split("+")[-1])  # 맨 뒤 pos정보만 사용
            head_list.append(int(raw_example.head))
            dep_list.append(raw_example.dep)
            prev_raw_example = raw_example
            prev_SENT_ID = SENT_ID

        encoded: BatchEncoding = tokenizer.encode_plus(" ".join(token_list),
                                                       max_length=self.args.model.seq_len,
                                                       truncation=TruncationStrategy.LONGEST_FIRST,
                                                       padding=PaddingStrategy.MAX_LENGTH)
        logger.debug(hr())
        logger.debug(f"encoded.tokens()        = {encoded.tokens()}")
        for key in encoded.keys():
            logger.debug(f"encoded[{key:14s}] = {encoded[key]}")

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
        bpe_head_mask.extend([0] * (self.args.model.seq_len - len(bpe_head_mask)))
        bpe_tail_mask.extend([0] * (self.args.model.seq_len - len(bpe_tail_mask)))
        head_ids.extend([-1] * (self.args.model.seq_len - len(head_ids)))
        dep_ids.extend([-1] * (self.args.model.seq_len - len(dep_ids)))
        pos_ids.extend([-1] * (self.args.model.seq_len - len(pos_ids)))

        encoded_example = DPEncodedExample(
            idx=prev_SENT_ID,
            raw=prev_raw_example,
            encoded=encoded,
            bpe_head_mask=bpe_head_mask,
            bpe_tail_mask=bpe_tail_mask,
            head_ids=head_ids,
            dep_ids=dep_ids,
            pos_ids=pos_ids,
        )
        encoded_examples.append(encoded_example)
        logger.debug(hr())
        logger.debug(f"bpe_head_mask           = {bpe_head_mask}")
        logger.debug(f"bpe_tail_mask           = {bpe_tail_mask}")
        logger.debug(f"head_ids                = {head_ids}")
        logger.debug(f"dep_ids                 = {dep_ids}")
        logger.debug(f"pos_ids                 = {pos_ids}")
        logger.debug("")

        logger.info(hr())
        for encoded_example in encoded_examples[:self.args.data.num_check]:
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

    def encoded_examples_to_batch(self, examples: List[DPEncodedExample]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        pos_padding_idx = len(self.get_pos_labels())
        first = examples[0]
        batch = {"example_ids": torch.tensor([ex.idx for ex in examples], dtype=torch.long)}

        # 2. build inputs : packing tensors
        # 나는 밥을 먹는다. => [CLS] 나 ##는 밥 ##을 먹 ##는 ##다 . [SEP]
        # input_id : [2, 717, 2259, 1127, 2069, 1059, 2259, 2062, 18, 3, 0, 0, ...]
        # bpe_head_mask : [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, ...] (indicate word start (head) idx)
        for k, v in first.encoded.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([ex.encoded[k] for ex in examples])
                else:
                    batch[k] = torch.tensor([ex.encoded[k] for ex in examples], dtype=torch.long)
        batch["bpe_head_mask"] = torch.tensor([ex.bpe_head_mask for ex in examples],
                                              dtype=torch.long if type(first.bpe_head_mask[0]) is int else torch.float)
        batch["bpe_tail_mask"] = torch.tensor([ex.bpe_tail_mask for ex in examples],
                                              dtype=torch.long if type(first.bpe_tail_mask[0]) is int else torch.float)

        # 3. token_to_words : set in-batch max_word_length
        max_word_length = max(torch.sum(batch["bpe_head_mask"], dim=1)).item()
        batch["max_word_length"] = torch.tensor(max_word_length, dtype=torch.long)

        # 3. token_to_words : placeholders
        batch["head_ids"] = torch.zeros(batch_size, max_word_length, dtype=torch.long)
        batch["type_ids"] = torch.zeros(batch_size, max_word_length, dtype=torch.long)
        batch["pos_ids"] = torch.zeros(batch_size, max_word_length + 1, dtype=torch.long)
        batch["mask_e"] = torch.zeros(batch_size, max_word_length + 1, dtype=torch.long)
        # 3. token_to_words : head_ids, type_ids, pos_ids, mask_e, mask_d
        for batch_id in range(batch_size):
            example = examples[batch_id]
            bpe_head_mask = example.bpe_head_mask
            token_head_ids = example.head_ids
            token_type_ids = example.dep_ids
            token_pos_ids = example.pos_ids
            # head_id : [1, 3, 5] (prediction candidates)
            # token_head_ids : [-1, 3, -1, 3, -1, 0, -1, -1, -1, .-1, ...] (ground truth head ids)
            head_id = [i for i, token in enumerate(example.bpe_head_mask) if token == 1]
            word_length = len(head_id)
            head_id.extend([0] * (max_word_length - word_length))
            batch["head_ids"][batch_id] = torch.tensor(token_head_ids, dtype=torch.long)[head_id]
            batch["type_ids"][batch_id] = torch.tensor(token_type_ids, dtype=torch.long)[head_id]
            batch["pos_ids"][batch_id][0] = torch.tensor(pos_padding_idx)
            batch["pos_ids"][batch_id][1:] = torch.tensor(token_pos_ids, dtype=torch.long)[head_id]
            batch["pos_ids"][batch_id][int(torch.sum(torch.tensor(bpe_head_mask, dtype=torch.long))) + 1:] = torch.tensor(pos_padding_idx)
            batch["mask_e"][batch_id] = torch.LongTensor([1] * (word_length + 1) + [0] * (max_word_length - word_length))
        batch["mask_d"] = batch["mask_e"][:, 1:]

        # 4. pack everything
        return batch


class DPDataset(Dataset):
    def __init__(self, split: str, tokenizer: PreTrainedTokenizerFast, corpus: DPCorpus):
        self.corpus: DPCorpus = corpus
        examples: List[DPRawExample] = self.corpus.read_raw_examples(split)
        self.dep_labels: List[str] = self.corpus.get_dep_labels()
        self.pos_labels: List[str] = self.corpus.get_pos_labels()
        self._dep_label_to_id: Dict[str, int] = {label: i for i, label in enumerate(self.dep_labels)}
        self._pos_label_to_id: Dict[str, int] = {label: i for i, label in enumerate(self.pos_labels)}
        self._id_to_dep_label: Dict[int, str] = {i: label for i, label in enumerate(self.dep_labels)}
        self._id_to_pos_label: Dict[int, str] = {i: label for i, label in enumerate(self.pos_labels)}
        self.features: List[DPEncodedExample] = self.corpus.raw_examples_to_encoded_examples(
            examples, tokenizer, pos_label_list=self.pos_labels, dep_label_list=self.dep_labels)

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
