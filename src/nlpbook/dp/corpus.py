from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import List, Optional, Dict, ClassVar

import pandas as pd
import torch
import typer
from dataclasses_json import DataClassJsonMixin
from sklearn.metrics import classification_report
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from chrisbase.data import AppTyper, ProjectEnv, InputOption, FileOption, IOArguments, OutputOption, JobTimer, FileStreamer, OptionData
from chrisbase.io import hr, LoggingFormat
from chrisbase.util import mute_tqdm_cls, LF
from chrisbase.util import to_dataframe
from nlpbook.arguments import TesterArguments, TrainerArguments
from nlpbook.metrics import DPResult, DP_UAS_MacroF1, DP_LAS_MacroF1, DP_UAS_MicroF1, DP_LAS_MicroF1

logger = logging.getLogger(__name__)
main = AppTyper()


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


@dataclass
class DPParsedExample(DataClassJsonMixin):
    column_names: ClassVar[List[str]] = ["id", "form", "lemma", "pos", "head", "label"]
    example_id: str = field(default_factory=str)
    sentence: str = field(default_factory=str)
    words: List[Dict[str, str | int]] = field(default_factory=list)

    @classmethod
    def from_tsv(cls, tsv: str):
        meta = [x.split('\t') for x in tsv.strip().splitlines() if x.startswith('#')][-1]
        words = ([["0", "<ROOT>", "<ROOT>", "<ROOT>", "-1", "<ROOT>"]] +
                 [x.split('\t') for x in tsv.strip().splitlines() if not x.startswith('#')])
        words = [dict(zip(cls.column_names, row)) for row in words]
        for word in words:
            word["id"] = int(word["id"])
            word["head"] = int(word["head"])
        example_id = re.sub(r"^##+", "", meta[0]).strip()
        sentence = meta[-1]

        return cls(example_id=example_id, sentence=sentence, words=words)


class EvaluateApp:
    app = AppTyper()
    label_names = DPCorpus.get_dep_labels()
    label_ids = [i for i, _ in enumerate(label_names)]
    label_to_id = {label: i for i, label in enumerate(label_names)}
    id_to_label = {i: label for i, label in enumerate(label_names)}

    dp_full_pattern = re.compile("-([0-9]+)(/([A-Z]{1,3}(_[A-Z]{1,3})?))?")

    @classmethod
    def typer(cls) -> typer.Typer:

        @dataclass
        class EvaluateArguments(IOArguments):
            refer: InputOption = field()

            def __post_init__(self):
                super().__post_init__()

            def dataframe(self, columns=None) -> pd.DataFrame:
                if not columns:
                    columns = [self.data_type, "value"]
                return pd.concat([
                    super().dataframe(columns=columns),
                    to_dataframe(columns=columns, raw=self.refer, data_prefix="refer", data_exclude=["file", "table", "index"]),
                    to_dataframe(columns=columns, raw=self.refer.file, data_prefix="refer.file") if self.refer.file else None,
                    to_dataframe(columns=columns, raw=self.refer.table, data_prefix="refer.table") if self.refer.table else None,
                    to_dataframe(columns=columns, raw=self.refer.index, data_prefix="refer.index") if self.refer.index else None,
                ]).reset_index(drop=True)

        def to_dp_result_v1(words: List[str]) -> DPResult:
            heads = [-1] * len(words)
            types = [-1] * len(words)
            for wi, word in enumerate(words):
                m = cls.dp_full_pattern.search(word)
                if m:
                    head, dep = m.group(1), m.group(3)
                    dep_id = cls.label_to_id.get(dep, 0) if dep else 0
                    head_id = int(head)
                    heads[wi] = head_id
                    types[wi] = dep_id
                else:
                    heads[wi] = 0
                    types[wi] = 0
            result = DPResult(torch.tensor(heads), torch.tensor(types))
            assert result.heads.shape == result.types.shape, f"result.heads.shape != result.types.shape: {result.heads.shape} != {result.types.shape}"
            return result

        @cls.app.callback(invoke_without_command=True)
        def evaluate(
                ctx: typer.Context,
                # env
                project: str = typer.Option(default="DeepKNLU"),
                job_name: str = typer.Option(default="evaluate"),
                output_home: str = typer.Option(default="output"),
                logging_file: str = typer.Option(default="logging.out"),
                debugging: bool = typer.Option(default=False),
                verbose: int = typer.Option(default=3),
                # data
                input_inter: int = typer.Option(default=5000),
                input_file_home: str = typer.Option(default="data"),
                input_file_name: str = typer.Option(default="klue-dp-pred/infer_klue_dp-v1.3.0.pred"),
                refer_file_home: str = typer.Option(default="data"),
                refer_file_name: str = typer.Option(default="klue-dp/klue-dp-v1.1_dev.seq-v1.3.tsv"),
                output_file_home: str = typer.Option(default="data"),
                output_file_name: str = typer.Option(default="klue-dp-pred/infer_klue_dp-v1.0.0.eval"),
        ):
            if ctx.invoked_subcommand is not None:
                return
            env = ProjectEnv(
                project=project,
                job_name=job_name,
                debugging=debugging,
                output_home=output_home,
                logging_file=logging_file,
                msg_level=logging.DEBUG if debugging else logging.INFO,
                msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
            )
            input_opt = InputOption(
                inter=input_inter,
                file=FileOption(
                    home=input_file_home,
                    name=input_file_name,
                    mode="r",
                    strict=True,
                ),
            )
            refer_opt = InputOption(
                file=FileOption(
                    home=refer_file_home,
                    name=refer_file_name,
                    mode="r",
                    strict=True,
                ),
            )
            output_opt = OutputOption(
                file=FileOption(
                    home=output_file_home,
                    name=output_file_name,
                    mode="w",
                    strict=True,
                ),
            )
            args = EvaluateArguments(
                env=env,
                input=input_opt,
                refer=refer_opt,
                output=output_opt,
            )
            tqdm = mute_tqdm_cls()
            assert args.input.file, "input.file is required"
            assert args.refer.file, "refer.file is required"
            assert args.output.file, "output.file is required"

            if verbose < 1:
                logging.getLogger("chrisbase.data").setLevel(logging.WARNING)
            with (
                JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}",
                         rt=1, rb=1, rc='=', verbose=verbose > 1, args=args if debugging or verbose > 2 else None),
                FileStreamer(args.input.file) as input_file,
                FileStreamer(args.refer.file) as refer_file,
                FileStreamer(args.output.file) as output_file,
            ):
                input_items = [x.strip() for x in input_file.path.read_text().split("Dependency Relations: ") if len(x.strip()) > 0]
                refer_items = [x.replace("<BR>", "\n").strip() for x in [x.split("Dependency Relations: ")[1] for x in refer_file] if len(x.strip()) > 0]
                logger.info(f"Load {len(input_items)} items from [{input_file.opt}]")
                logger.info(f"Load {len(refer_items)} items from [{refer_file.opt}]")
                assert len(input_items) == len(refer_items), f"Length of input_items and refer_items are different: {len(input_items)} != {len(refer_items)}"
                progress, interval = (
                    tqdm(zip(input_items, refer_items), total=len(input_items), unit="item", pre="*", desc="evaluating"),
                    args.input.inter,
                )

                golds, preds = [], []
                gold_heads, pred_heads = [], []
                gold_types, pred_types = [], []
                num_shorter, num_longer = 0, 0
                for i, (a, b) in enumerate(progress):
                    if i > 0 and i % interval == 0:
                        logger.info(progress)
                    pred_words = a.strip().splitlines()[0].split("▁")
                    gold_words = b.strip().splitlines()[0].split("▁")
                    if len(pred_words) < len(gold_words):
                        num_shorter += 1
                        continue
                        # logger.warning(f"[{i:04d}] Shorter pred_words({len(pred_words)}): {pred_words}")
                        # logger.warning(f"[{i:04d}]         gold_words({len(gold_words)}): {gold_words}")
                        # pred_words = pred_words + (
                        #         [pred_words[-1]] * (len(gold_words) - len(pred_words))
                        # )
                        # logger.warning(f"[{i:04d}]      -> pred_words({len(pred_words)}): {pred_words}")
                    if len(pred_words) > len(gold_words):
                        num_longer += 1
                        continue
                        # logger.warning(f"[{i:04d}]  Longer pred_words({len(pred_words)}): {pred_words}")
                        # logger.warning(f"[{i:04d}]         gold_words({len(gold_words)}): {gold_words}")
                        # pred_words = pred_words[:len(gold_words)]
                        # logger.warning(f"[{i:04d}]      -> pred_words({len(pred_words)}): {pred_words}")
                    assert len(pred_words) == len(gold_words), f"Length of pred_words and gold_words are different: {len(pred_words)} != {len(gold_words)}"
                    pred_res = to_dp_result_v1(pred_words)
                    gold_res = to_dp_result_v1(gold_words)
                    print(f"gold_res({'x'.join(map(str, gold_res.heads.shape))}): {gold_res}")
                    print(f"test_res({'x'.join(map(str, pred_res.heads.shape))}): {pred_res}")
                    print()
                    assert gold_res.heads.shape == pred_res.heads.shape, f"gold_res.heads.shape != pred_res.heads.shape: {gold_res.heads.shape} != {pred_res.heads.shape}"
                    assert gold_res.types.shape == pred_res.types.shape, f"gold_res.types.shape != pred_res.types.shape: {gold_res.types.shape} != {pred_res.types.shape}"
                    golds.append(gold_res)
                    preds.append(pred_res)
                    gold_types.extend(gold_res.types.tolist())
                    pred_types.extend(pred_res.types.tolist())
                    gold_heads.extend(gold_res.heads.tolist())
                    pred_heads.extend(pred_res.heads.tolist())
                logger.info(progress)
                assert len(golds) == len(preds), f"Length of golds and preds are different: {len(golds)} != {len(preds)}"

                res1 = classification_report(gold_types, pred_types, labels=cls.label_ids, target_names=cls.label_names, digits=4, zero_division=1)
                logger.info(hr(c='-'))
                for line in res1.splitlines():
                    logger.info(line)
                res2 = classification_report(gold_heads, pred_heads, digits=4, zero_division=1)
                logger.info(hr(c='-'))
                for line in res2.splitlines():
                    logger.info(line)
                logger.info(hr(c='-'))

                DP_UAS_MacroF1.reset()
                DP_LAS_MacroF1.reset()
                DP_UAS_MicroF1.reset()
                DP_LAS_MicroF1.reset()
                DP_UAS_MacroF1.update(preds, golds)
                DP_LAS_MacroF1.update(preds, golds)
                DP_UAS_MicroF1.update(preds, golds)
                DP_LAS_MicroF1.update(preds, golds)
                logger.info(f"#evaluated_cases: #{len(preds)}")
                logger.info(f"- DP UASa = {DP_UAS_MacroF1.compute():.4f}")
                logger.info(f"- DP LASa = {DP_LAS_MacroF1.compute():.4f}")
                logger.info(f"- DP UASi = {DP_UAS_MicroF1.compute():.4f}")
                logger.info(f"- DP LASi = {DP_LAS_MicroF1.compute():.4f}")

        return cls.app


class ConvertApp:
    app = AppTyper()

    @classmethod
    def typer(cls) -> typer.Typer:

        @dataclass
        class ConvertOption(OptionData):
            level_major: int = field()
            level_minor: int = field()

        @dataclass
        class ConvertArguments(IOArguments):
            convert: ConvertOption = field()

            def __post_init__(self):
                super().__post_init__()

            def dataframe(self, columns=None) -> pd.DataFrame:
                if not columns:
                    columns = [self.data_type, "value"]
                return pd.concat([
                    super().dataframe(columns=columns),
                    to_dataframe(columns=columns, raw=self.convert, data_prefix="convert"),
                ]).reset_index(drop=True)

        def to_str(s: StringIO):
            s.seek(0)
            return s.read().replace("\n", "<BR>")

        def to_seq1(example: DPParsedExample, level: int):
            assert level < 2, f"Unsupported seq1 level: {level}"
            with StringIO() as s:
                print("Task: Dependency Parsing", file=s)
                print('', file=s)
                print(f"Input: {' '.join(word['form'] for word in example.words[1:])}", file=s)
                if level < 1:
                    return to_str(s)

                units1 = []
                for word in example.words[1:]:
                    ls = word["lemma"].split(" ")
                    ps = word["pos"].split("+")
                    unit1 = f"{word['id']}/{word['form']}/{len(word['form'])}/{word['lemma']}/{ls[0]}:{ps[0]}/" + (f"{ls[-1]}:{ps[-1]}" if len(ps) > 1 else "NONE")
                    units1.append(unit1)
                print(f"Lemmas: {LF.join(units1)}", end='', file=s)
                if level < 2:
                    return to_str(s)

        def to_seq2(example: DPParsedExample, level: int):
            assert level < 4, f"Unsupported seq2 level: {level}"
            with (StringIO() as s):
                units2 = []
                for word in example.words[1:]:
                    if level < 1:
                        unit2 = f"{word['head']}/{word['label']}"
                    elif level < 2:
                        unit2 = f"{example.words[word['head']]['form']}-{word['head']}/{word['label']}"
                    elif level < 3:
                        d = f"{word['form']}-{word['id']}"
                        h = f"{example.words[word['head']]['form']}-{word['head']}"
                        unit2 = f"{word['label']}({d}, {h})"
                    elif level < 4:
                        unit2 = f"({word['id']}/{len(word['form'])}, {word['head']}, {word['label']})"
                    units2.append(unit2)
                print(f"Dependency Relations: {'▁'.join(units2)}", file=s)
                print(f"Word Count: {len(example.words) - 1}", file=s)
                return to_str(s)

        @cls.app.callback(invoke_without_command=True)
        def convert(
                ctx: typer.Context,
                # env
                project: str = typer.Option(default="DeepKNLU"),
                job_name: str = typer.Option(default="convert"),
                output_home: str = typer.Option(default="output"),
                logging_file: str = typer.Option(default="logging.out"),
                debugging: bool = typer.Option(default=False),
                verbose: int = typer.Option(default=0),
                # data
                input_inter: int = typer.Option(default=5000),
                input_file_home: str = typer.Option(default="data"),
                input_file_name: str = typer.Option(default="klue-dp-mini/klue-dp-v1.1_dev.tsv"),
                output_file_home: str = typer.Option(default="data"),
                output_file_name: str = typer.Option(default="klue-dp-mini/klue-dp-v1.1_dev.tsv"),
                # convert
                level_major: int = typer.Option(default=0),
                level_minor: int = typer.Option(default=0),
        ):
            if ctx.invoked_subcommand is not None:
                return
            env = ProjectEnv(
                project=project,
                job_name=job_name,
                debugging=debugging,
                output_home=output_home,
                logging_file=logging_file,
                msg_level=logging.DEBUG if debugging else logging.INFO,
                msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
            )
            input_opt = InputOption(
                inter=input_inter,
                file=FileOption(
                    home=input_file_home,
                    name=input_file_name,
                    mode="r",
                    strict=True,
                ),
            )
            output_file_name = Path(output_file_name)
            output_opt = OutputOption(
                file=FileOption(
                    home=output_file_home,
                    name=output_file_name.with_suffix(f".seq-v{level_major}.{level_minor}{output_file_name.suffix}"),
                    mode="w",
                    strict=True,
                ),
            )
            convert_opt = ConvertOption(
                level_major=level_major,
                level_minor=level_minor,
            )
            args = ConvertArguments(
                env=env,
                input=input_opt,
                output=output_opt,
                convert=convert_opt,
            )
            tqdm = mute_tqdm_cls()
            assert args.input.file, "input.file is required"
            assert args.output.file, "output.file is required"

            if verbose < 1:
                logging.getLogger("chrisbase.data").setLevel(logging.WARNING)
            with (
                JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}",
                         rt=1, rb=1, rc='=', verbose=verbose > 1, args=args if debugging or verbose > 2 else None),
                FileStreamer(args.input.file) as input_file,
                FileStreamer(args.output.file) as output_file,
            ):
                input_file.path.read_text()
                input_chunks = [x for x in input_file.path.read_text().split("\n\n") if len(x.strip()) > 0]
                logger.info(f"Load {len(input_chunks)} sentences from [{input_file.opt}]")
                progress, interval = (
                    tqdm(input_chunks, total=len(input_chunks), unit="sent", pre="*", desc="converting"),
                    args.input.inter,
                )
                num_output = 0
                for i, x in enumerate(progress):
                    if i > 0 and i % interval == 0:
                        logger.info(progress)
                    example = DPParsedExample.from_tsv(x)
                    seq1 = to_seq1(example, args.convert.level_major)
                    seq2 = to_seq2(example, args.convert.level_minor)
                    output_file.fp.write(seq1 + "\t" + seq2 + "\n")
                    num_output += 1
                logger.info(progress)
                logger.info(f"Saved {num_output} sequence pairs to [{output_file.opt}]")

        return cls.app


main.add_typer(ConvertApp.typer(), name="convert")
main.add_typer(EvaluateApp.typer(), name="evaluate")

if __name__ == "__main__":
    main()
