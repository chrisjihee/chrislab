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
from transformers import CharSpan
from transformers import PreTrainedTokenizerFast, BatchEncoding
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from chrisbase.data import AppTyper, ProjectEnv, InputOption, FileOption, IOArguments, OutputOption, JobTimer, FileStreamer, OptionData, ResultData
from chrisbase.io import hr, LoggingFormat, file_size, setup_unit_logger
from chrisbase.io import make_parent_dir, merge_dicts
from chrisbase.util import mute_tqdm_cls, LF, HT, NO
from chrisbase.util import to_dataframe
from nlpbook.arguments import MLArguments
from nlpbook.metrics import NER_Char_MacroF1, NER_Entity_MacroF1

logger = logging.getLogger(__name__)
setup_unit_logger(fmt=LoggingFormat.CHECK_24)


@dataclass
class EntityInText(DataClassJsonMixin):
    pattern: ClassVar[re.Pattern] = re.compile('<(?P<text>[^<>]+?):(?P<label>[A-Z]{2,3})>')
    text: str
    label: str
    offset: tuple[int, int]

    @staticmethod
    def from_match(m: re.Match, s: str) -> tuple["EntityInText", str]:
        g = m.groupdict()
        x = g['text']
        y = g['label']
        z = (m.start(), m.start() + len(x))
        e = EntityInText(text=x, label=y, offset=z)
        s = s[:m.start()] + x + s[m.end():]
        return e, s

    def to_offset_lable_dict(self) -> Dict[int, str]:
        offset_list = [(self.offset[0], f"B-{self.label}")]
        for i in range(self.offset[0] + 1, self.offset[1]):
            offset_list.append((i, f"I-{self.label}"))
        return dict(offset_list)


@dataclass
class NERTaggedExample(DataClassJsonMixin):
    example_id: str = field(default_factory=str)
    origin: str = field(default_factory=str)
    tagged: str = field(default_factory=str)

    @classmethod
    def from_tsv(cls, tsv: str):
        lines = tsv.strip().splitlines()
        meta = [x.split('\t') for x in lines if x.startswith('#')][-1]
        chars = [x.split('\t') for x in lines if not x.startswith('#')]
        example_id = re.sub(r"^##+", "", meta[0]).strip()
        tagged = meta[1].strip()
        origin = ''.join(x[0] for x in chars)
        return cls(example_id=example_id, origin=origin, tagged=tagged)


@dataclass
class NERParsedExample(DataClassJsonMixin):
    origin: str = field(default_factory=str)
    entity_list: List[EntityInText] = field(default_factory=list)
    character_list: List[tuple[str, str]] = field(default_factory=list)

    def get_offset_label_dict(self):
        return {i: y for i, (_, y) in enumerate(self.character_list)}

    def to_tagged_text(self, entity_form=lambda e: f"<{e.text}:{e.label}>"):
        self.entity_list.sort(key=lambda x: x.offset[0])
        cursor = 0
        tagged_text = ""
        for e in self.entity_list:
            tagged_text += self.origin[cursor: e.offset[0]] + entity_form(e)
            cursor = e.offset[1]
        tagged_text += self.origin[cursor:]
        return tagged_text

    @classmethod
    def from_tagged(cls, origin: str, tagged: str, debug: bool = False) -> Optional["NERParsedExample"]:
        entity_list: List[EntityInText] = []
        if debug:
            logging.debug(f"* origin: {origin}")
            logging.debug(f"  tagged: {tagged}")
        restored = tagged[:]
        no_problem = True
        offset_labels = {i: "O" for i in range(len(origin))}
        while True:
            match: re.Match = EntityInText.pattern.search(restored)
            if not match:
                break
            entity, restored = EntityInText.from_match(match, restored)
            extracted = origin[entity.offset[0]:entity.offset[1]]
            if entity.text == extracted:
                entity_list.append(entity)
                offset_labels = merge_dicts(offset_labels, entity.to_offset_lable_dict())
            else:
                no_problem = False
            if debug:
                logging.debug(f"  = {entity} -> {extracted}")
                logging.debug(f"    {offset_labels}")
        if debug:
            logging.debug(f"  --------------------")
        character_list = [(origin[i], offset_labels[i]) for i in range(len(origin))]
        if restored != origin:
            no_problem = False
        return cls(origin=origin,
                   entity_list=entity_list,
                   character_list=character_list) if no_problem else None


@dataclass
class NEREncodedExample:
    idx: int
    raw: NERParsedExample
    encoded: BatchEncoding
    label_ids: Optional[List[int]] = None


class NERCorpus:
    def __init__(self, args: MLArguments):
        self.args = args

    @property
    def num_labels(self) -> int:
        return len(self.get_labels())

    @classmethod
    def get_labels_from_data(cls,
                             data_path: str | Path = "data/klue-ner/klue-ner-v1.1_dev.jsonl",
                             label_path: str | Path = "data/klue-ner/label_map.txt") -> List[str]:
        label_path = make_parent_dir(label_path).absolute()
        data_path = Path(data_path).absolute()
        assert data_path.exists() and data_path.is_file() or label_path.exists() and label_path.is_file(), f"No data_path or label_path: {data_path}, {label_path}"
        if label_path.exists():
            labels = label_path.read_text().splitlines()
            logger.info(f"Loaded {len(labels)} labels from {label_path}")
        else:
            logger.info(f"Extracting labels from {data_path}")
            ner_tags = []
            with data_path.open() as inp:
                for line in inp.readlines():
                    for x in NERParsedExample.from_json(line).entity_list:
                        if x.label not in ner_tags:
                            ner_tags.append(x.label)
            ner_tags = sorted(ner_tags)
            b_tags = [f"B-{ner_tag}" for ner_tag in ner_tags]
            i_tags = [f"I-{ner_tag}" for ner_tag in ner_tags]
            labels = ["O"] + b_tags + i_tags
            logger.info(f"Saved {len(labels)} labels to {label_path}")
            with label_path.open("w") as f:
                f.writelines([x + "\n" for x in labels])
        return labels

    def get_labels(self) -> List[str]:
        label_path = make_parent_dir(self.args.env.output_home / "label_map.txt")
        train_data_path = self.args.data.home / self.args.data.name / self.args.data.files.train if self.args.data.files.train else None
        valid_data_path = self.args.data.home / self.args.data.name / self.args.data.files.valid if self.args.data.files.valid else None
        test_data_path = self.args.data.home / self.args.data.name / self.args.data.files.test if self.args.data.files.test else None
        train_data_path = train_data_path if train_data_path and train_data_path.exists() else None
        valid_data_path = valid_data_path if valid_data_path and valid_data_path.exists() else None
        test_data_path = test_data_path if test_data_path and test_data_path.exists() else None
        data_path = train_data_path or valid_data_path or test_data_path
        return self.get_labels_from_data(data_path=data_path, label_path=label_path)

    def read_raw_examples(self, split: str) -> List[NERParsedExample]:
        assert self.args.data.home, f"No data_home: {self.args.data.home}"
        assert self.args.data.name, f"No data_name: {self.args.data.name}"
        data_file_dict: dict = self.args.data.files.to_dict()
        assert split in data_file_dict, f"No '{split}' split in data_file: should be one of {list(data_file_dict.keys())}"
        assert data_file_dict[split], f"No data_file for '{split}' split: {self.args.data.files}"
        data_path: Path = Path(self.args.data.home) / self.args.data.name / data_file_dict[split]
        assert data_path.exists() and data_path.is_file(), f"No data_text_path: {data_path}"
        logger.info(f"Creating features from {data_path}")

        examples = []
        with data_path.open(encoding="utf-8") as inp:
            for line in inp.readlines():
                examples.append(NERParsedExample.from_json(line))
        logger.info(f"Loaded {len(examples)} examples from {data_path}")
        return examples

    @staticmethod
    def _decide_span_label(span: CharSpan, offset_to_label: Dict[int, str]):
        for x in [offset_to_label[i] for i in range(span.start, span.end)]:
            if x.startswith("B-") or x.startswith("I-"):
                return x
        return "O"

    def raw_examples_to_encoded_examples(
            self,
            raw_examples: List[NERParsedExample],
            tokenizer: PreTrainedTokenizerFast,
            label_list: List[str],
    ) -> List[NEREncodedExample]:
        label_to_id: Dict[str, int] = {label: i for i, label in enumerate(label_list)}
        id_to_label: Dict[int, str] = {i: label for i, label in enumerate(label_list)}
        logger.debug(f"label_to_id = {label_to_id}")
        logger.debug(f"id_to_label = {id_to_label}")

        encoded_examples: List[NEREncodedExample] = []
        for idx, raw_example in enumerate(raw_examples):
            raw_example: NERParsedExample = raw_example
            offset_to_label: Dict[int, str] = raw_example.get_offset_label_dict()
            logger.debug(hr())
            logger.debug(f"offset_to_label = {offset_to_label}")
            encoded: BatchEncoding = tokenizer.encode_plus(raw_example.origin,
                                                           max_length=self.args.model.seq_len,
                                                           truncation=TruncationStrategy.LONGEST_FIRST,
                                                           padding=PaddingStrategy.MAX_LENGTH)
            encoded_tokens: List[str] = encoded.tokens()
            logger.debug(hr())
            logger.debug(f"encoded.tokens()           = {encoded.tokens()}")
            for key in encoded.keys():
                logger.debug(f"encoded[{key:14s}]    = {encoded[key]}")

            logger.debug(hr())
            label_list: List[str] = []
            for token_id in range(self.args.model.seq_len):
                token_repr: str = encoded_tokens[token_id]
                token_span: CharSpan = encoded.token_to_chars(token_id)
                if token_span:
                    token_label = self._decide_span_label(token_span, offset_to_label)
                    label_list.append(token_label)
                    token_sstr = raw_example.origin[token_span.start:token_span.end]
                    logger.debug('\t'.join(map(str, [token_id, token_repr, token_span, token_sstr, token_label])))
                else:
                    label_list.append('O')
                    logger.debug('\t'.join(map(str, [token_id, token_repr, token_span])))
            label_ids: List[int] = [label_to_id[label] for label in label_list]
            encoded_example = NEREncodedExample(idx=idx, raw=raw_example, encoded=encoded, label_ids=label_ids)
            encoded_examples.append(encoded_example)
            logger.debug(hr())
            logger.debug(f"label_list                = {label_list}")
            logger.debug(f"label_ids                 = {label_ids}")
            logger.debug(hr())
            logger.debug(f"encoded_example.idx       = {encoded_example.idx}")
            logger.debug(f"encoded_example.raw       = {encoded_example.raw}")
            logger.debug(f"encoded_example.encoded   = {encoded_example.encoded}")
            logger.debug(f"encoded_example.label_ids = {encoded_example.label_ids}")

        logger.info(hr())
        for encoded_example in encoded_examples[:self.args.data.num_check]:
            logger.info("  === [Example %d] ===" % encoded_example.idx)
            logger.info("  = sentence   : %s" % encoded_example.raw.origin)
            logger.info("  = characters : %s" % " | ".join(f"{x}/{y}" for x, y in encoded_example.raw.character_list))
            logger.info("  = tokens     : %s" % " ".join(encoded_example.encoded.tokens()))
            logger.info("  = labels     : %s" % " ".join([id_to_label[x] for x in encoded_example.label_ids]))
            logger.info("  === ")

        logger.info(f"Converted {len(raw_examples)} raw examples to {len(encoded_examples)} encoded examples")
        return encoded_examples

    @staticmethod
    def encoded_examples_to_batch(examples: List[NEREncodedExample]) -> Dict[str, torch.Tensor]:
        first = examples[0]
        batch = {}
        for k, v in first.encoded.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([ex.encoded[k] for ex in examples])
                else:
                    batch[k] = torch.tensor([ex.encoded[k] for ex in examples], dtype=torch.long)
        batch["labels"] = torch.tensor([ex.label_ids for ex in examples],
                                       dtype=torch.long if type(first.label_ids[0]) is int else torch.float)
        batch["example_ids"] = torch.tensor([ex.idx for ex in examples], dtype=torch.int)
        return batch


class NERDataset(Dataset):
    def __init__(self, split: str, tokenizer: PreTrainedTokenizerFast, corpus: NERCorpus):
        self.corpus: NERCorpus = corpus
        examples: List[NERParsedExample] = self.corpus.read_raw_examples(split)
        self.label_list: List[str] = self.corpus.get_labels()
        self._label_to_id: Dict[str, int] = {label: i for i, label in enumerate(self.label_list)}
        self._id_to_label: Dict[int, str] = {i: label for i, label in enumerate(self.label_list)}
        self.features: List[NEREncodedExample] = self.corpus.raw_examples_to_encoded_examples(
            examples, tokenizer, label_list=self.label_list)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i) -> NEREncodedExample:
        return self.features[i]

    def get_labels(self) -> List[str]:
        return self.label_list

    def label_to_id(self, label: str) -> int:
        return self._label_to_id[label]

    def id_to_label(self, label_id: int) -> str:
        return self._id_to_label[label_id]


class NERCorpusConverter:
    @classmethod
    def convert_from_kmou_format(cls, infile: str | Path, outfile: str | Path, debug: bool = False):
        with Path(infile).open(encoding="utf-8") as inp, Path(outfile).open("w", encoding="utf-8") as out:
            for line in inp.readlines():
                origin, tagged = line.strip().split("\u241E")
                parsed: Optional[NERParsedExample] = NERParsedExample.from_tagged(origin, tagged, debug=debug)
                if parsed:
                    out.write(parsed.to_json(ensure_ascii=False) + "\n")

    @classmethod
    def convert_from_klue_format(cls, infile: str | Path, outfile: str | Path, debug: bool = False):
        with Path(infile) as inp, Path(outfile).open("w", encoding="utf-8") as out:
            raw_text = inp.read_text(encoding="utf-8").strip()
            raw_docs = re.split(r"\n\t?\n", raw_text)
            for raw_doc in raw_docs:
                raw_lines = raw_doc.splitlines()
                num_header = 0
                for line in raw_lines:
                    if not line.startswith("##"):
                        break
                    num_header += 1
                head_lines = raw_lines[:num_header]
                body_lines = raw_lines[num_header:]

                origin = ''.join(x.split("\t")[0] for x in body_lines)
                tagged = head_lines[-1].split("\t")[1].strip()
                parsed: Optional[NERParsedExample] = NERParsedExample.from_tagged(origin, tagged, debug=debug)
                if parsed:
                    character_list_from_head = parsed.character_list
                    character_list_from_body = [tuple(x.split("\t")) for x in body_lines]
                    if character_list_from_head == character_list_from_body:
                        out.write(parsed.to_json(ensure_ascii=False) + "\n")
                    elif debug:
                        print(f"* origin: {origin}")
                        print(f"  tagged: {tagged}")
                        for a, b in zip(character_list_from_head, character_list_from_body):
                            if a != b:
                                print(f"  = {a[0]}:{a[1]} <=> {b[0]}:{b[1]}")
                        print(f"  ====================")


@dataclass
class EvaluateResult(ResultData):
    s2s_type: str
    file_answer: str
    file_predict: str
    cate_predict: str
    post_predict: str
    num_answer: int
    num_predict: int
    num_evaluate: int
    num_mismatched: int
    num_skipped: int
    num_shorter: int
    num_longer: int
    metric_F1c: float
    metric_F1e: float


class CLI:
    main = AppTyper()
    task = "Named Entity Recognition"
    LINE_SEP = "<LF>"
    EACH_SEP = "▁"
    INPUT_PROMPT = "Input: "
    LABEL_MAIN_PROMPT = f"{task} on Sentence: "
    LABEL_EACH_PROMPT = f"{task} on Character: "
    NER_LABELS = [
        "O",
        "B-PS", "I-PS",
        "B-LC", "I-LC",
        "B-OG", "I-OG",
        "B-DT", "I-DT",
        "B-TI", "I-TI",
        "B-QT", "I-QT",
    ]
    label_ids = [i for i, _ in enumerate(NER_LABELS)]
    label_to_id = {label: i for i, label in enumerate(NER_LABELS)}
    id_to_label = {i: label for i, label in enumerate(NER_LABELS)}
    seq2_regex = {  # fullmatch version
        # TODO: check pattern /* -> /+ or /
        'a': re.compile(r"(?P<label>[BIO](-[A-Z]{2,3})?)"),
        'b': re.compile(r"[^ ]*/*(?P<label>[BIO](-[A-Z]{2,3})?)"),
        'c': re.compile(r"(?P<index>[0-9]+)\(*[^ ]*/*(?P<label>[BIO](-[A-Z]{2,3})?)\)*"),
        'd': re.compile(r"(?P<label>[BIO](-[A-Z]{2,3})?)\(*(?P<index>[0-9]+)/*[^ ]*\)*"),
        'e': re.compile(r"(?P<index>[0-9]+)/*[^ ]*=*>*(?P<label>[BIO](-[A-Z]{2,3})?)"),
        'f': re.compile(r"(?P<index>[0-9]+)\(*[^ ]*/*(?P<label>[BIO](-[A-Z]{2,3})?)\)*"),
        'g': re.compile(r"(?P<label>[BIO](-[A-Z]{2,3})?)\(*(?P<index>[0-9]+)/*[^ ]*\)*"),
        'h': re.compile(r"(?P<index>[0-9]+)/*[^ ]*=*>*(?P<label>[BIO](-[A-Z]{2,3})?)"),
    }

    @classmethod
    def strip_label_prompt(cls, x: str):
        x = x.replace(cls.LINE_SEP, LF)
        x = x.replace(cls.LABEL_MAIN_PROMPT, NO)
        x = x.replace(cls.LABEL_EACH_PROMPT, NO)
        x = x.replace(cls.LABEL_MAIN_PROMPT.strip(), NO)
        x = x.replace(cls.LABEL_EACH_PROMPT.strip(), NO)
        x = x.strip()
        return x

    @dataclass
    class ConvertOption(OptionData):
        s2s_type: str = field()
        seq1_type: str = field(init=False)
        seq2_type: str = field(init=False)

        def __post_init__(self):
            self.seq1_type = self.s2s_type[:2]
            self.seq2_type = self.s2s_type[-1:]

    @dataclass
    class ConvertArguments(IOArguments):
        convert: "CLI.ConvertOption" = field()

        def __post_init__(self):
            super().__post_init__()

        def dataframe(self, columns=None) -> pd.DataFrame:
            if not columns:
                columns = [self.data_type, "value"]
            return pd.concat([
                super().dataframe(columns=columns),
                to_dataframe(columns=columns, raw=self.convert, data_prefix="convert"),
            ]).reset_index(drop=True)

        def to_seq_pairs(self, example: NERParsedExample):
            main_label = example.to_tagged_text(lambda e: f"<{e.text}:{e.label}>")
            sub_forms = []
            sub_labels = []
            for i, (c, t) in enumerate(example.character_list, start=1):
                if c.strip():
                    sub_forms.append(f'{i}/{c}')
                    if self.convert.seq2_type == 'a':
                        sub_labels.append(f"{t}")
                    elif self.convert.seq2_type == 'b':
                        sub_labels.append(f"{c}/{t}")
                    elif self.convert.seq2_type == 'c':
                        sub_labels.append(f"{i}({c}/{t})")
                    elif self.convert.seq2_type == 'd':
                        sub_labels.append(f"{t}({i}/{c})")
                    elif self.convert.seq2_type == 'e':
                        sub_labels.append(f"{i}/{c}=>{t}")
                    elif self.convert.seq2_type == 'f':
                        if t != 'O':
                            sub_labels.append(f"{i}({c}/{t})")
                    elif self.convert.seq2_type == 'g':
                        if t != 'O':
                            sub_labels.append(f"{t}({i}/{c})")
                    elif self.convert.seq2_type == 'h':
                        if t != 'O':
                            sub_labels.append(f"{i}/{c}=>{t}")
                    elif self.convert.seq2_type == 'm':
                        pass
                    else:
                        raise NotImplementedError(f"Unsupported convert: {self.convert}")

            with StringIO() as s:
                print(f"Task: {CLI.task}", file=s)
                print('', file=s)
                print(f"Input: {example.origin}", file=s)

                if self.convert.seq1_type == 'S0':
                    seq1 = [CLI.to_str(s)]

                elif self.convert.seq1_type in ('S1', 'C1'):
                    print(f"Forms: {LF.join(sub_forms)}", file=s)
                    if self.convert.seq1_type == 'S1':
                        seq1 = [CLI.to_str(s)]
                    elif self.convert.seq1_type == 'C1':
                        seq1 = [CLI.to_str(s) + f"Target: {form}" + CLI.LINE_SEP for form in sub_forms]
                    else:
                        raise NotImplementedError(f"Unsupported convert: {self.convert}")

                else:
                    raise NotImplementedError(f"Unsupported convert: {self.convert}")

            if self.convert.seq1_type.startswith('S'):
                with StringIO() as s:
                    if self.convert.seq2_type == 'm':
                        print(CLI.LABEL_MAIN_PROMPT + main_label, file=s)
                    else:
                        print(CLI.LABEL_MAIN_PROMPT + CLI.EACH_SEP.join(sub_labels), file=s)
                        print(f"Label Count: {len(sub_labels)}", file=s)
                    seq2 = [CLI.to_str(s)]
            else:
                if len(sub_labels) != len(seq1):
                    return []
                seq2 = [CLI.LABEL_EACH_PROMPT + label for label in sub_labels]

            if len(seq1) != len(seq2):
                return []
            return zip(seq1, seq2)

    @dataclass
    class EvaluateOption(OptionData):
        skip_longer: bool = field()
        skip_shorter: bool = field()

    @dataclass
    class EvaluateArguments(IOArguments):
        refer: InputOption = field()
        convert: "CLI.ConvertOption" = field()
        evaluate: "CLI.EvaluateOption" = field()

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
                to_dataframe(columns=columns, raw=self.convert, data_prefix="convert"),
            ]).reset_index(drop=True)

    @staticmethod
    def to_str(s: StringIO):
        s.seek(0)
        return s.read().replace(LF, CLI.LINE_SEP)

    @staticmethod
    @main.command()
    def convert(
            # env
            project: str = typer.Option(default="DeepKNLU"),
            output_home: str = typer.Option(default="output"),
            logging_file: str = typer.Option(default="logging.out"),
            debugging: bool = typer.Option(default=False),
            verbose: int = typer.Option(default=1),
            # data
            input_inter: int = typer.Option(default=5000),
            input_file_name: str = typer.Option(default="data/klue-ner/klue-ner-v1.1_dev.tsv"),
            output_file_name: str = typer.Option(default="data/klue-ner/klue-ner-v1.1_dev-s2s.tsv"),
            # convert
            s2s_type: str = typer.Option(default="S0a"),
    ):
        env = ProjectEnv(
            project=project,
            debugging=debugging,
            output_home=output_home,
            logging_file=logging_file,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_36 if debugging else LoggingFormat.CHECK_24,
        )
        input_opt = InputOption(
            inter=input_inter,
            file=FileOption(
                name=input_file_name,
                mode="r",
                strict=True,
            ),
        )
        output_file_name = Path(output_file_name)
        output_opt = OutputOption(
            file=FileOption(
                name=output_file_name.with_stem(f"{output_file_name.stem}={s2s_type}"),
                mode="w",
                strict=True,
            ),
        )
        convert_opt = CLI.ConvertOption(
            s2s_type=s2s_type,
        )
        args = CLI.ConvertArguments(
            env=env,
            input=input_opt,
            output=output_opt,
            convert=convert_opt,
        )
        tqdm = mute_tqdm_cls()
        assert args.input.file, "input.file is required"
        assert args.output.file, "output.file is required"

        with (
            JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}",
                     rt=1, rb=1, mb=1, rc='=', verbose=verbose > 0, args=args if debugging or verbose > 1 else None),
            FileStreamer(args.input.file) as input_file,
            FileStreamer(args.output.file) as output_file,
        ):
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
                example = NERTaggedExample.from_tsv(x)
                example = NERParsedExample.from_tagged(example.origin, example.tagged, debug=True)
                if not example:
                    continue
                for seq1, seq2 in args.to_seq_pairs(example):
                    output_file.fp.write(seq1 + HT + seq2 + LF)
                    num_output += 1
            logger.info(progress)
            logger.info(f"Saved {num_output} sequence pairs to [{output_file.opt}]")
            if file_size(output_file.path) == 0:
                logger.info(f"Remove empty output file: [{output_file.opt}]")
                output_file.path.unlink()

    @staticmethod
    def units_to_label_ids(pred_units: List[str], gold_units: List[str], sentence: str, convert: "CLI.ConvertOption"):
        def value_to_label(regex: re.Pattern, unit_value: str):
            num_mismatch = 0
            match: re.Match = regex.fullmatch(unit_value)  # search or match or fullmatch
            if match:
                group = match.groupdict()
                label = group['label']
                label_id = CLI.label_to_id.get(label, 0) if label else 0
            else:
                logger.warning(f"Not found: regex={regex}, string={unit_value}")
                label_id = 0
                num_mismatch += 1
            return label_id, num_mismatch

        def values_to_labels(regex: re.Pattern, unit_values: List[str]):
            every_label_ids = [-1] + [0 if len(x.strip()) > 0 else -1 for x in sentence]
            unit_indices = [i for i, x in enumerate(every_label_ids) if x >= 0]
            num_mismatch = 0
            for i, unit_value in zip(unit_indices, unit_values):
                match: re.Match = regex.fullmatch(unit_value)  # search or match or fullmatch
                if match:
                    group = match.groupdict()
                    label = group['label']
                    index = int(group['index']) if 'index' in group else i
                    label_id = CLI.label_to_id.get(label, 0) if label else 0
                    if index in unit_indices:
                        every_label_ids[index] = label_id
                else:
                    logger.warning(f"Not found: regex={regex}, string={unit_value}")
                    num_mismatch += 1
            valid_label_ids = [x for x in every_label_ids if x >= 0]
            return valid_label_ids, num_mismatch

        if convert.seq1_type.startswith('S'):
            if convert.seq2_type == 'm':
                pred_parsed: Optional[NERParsedExample] = NERParsedExample.from_tagged(sentence, pred_units[0])
                if not pred_parsed:
                    return None, None, 0
                gold_parsed: Optional[NERParsedExample] = NERParsedExample.from_tagged(sentence, gold_units[0])
                assert gold_parsed, f"Failed to parse: sentence={sentence}, gold={gold_units[0]}"
                pred_label_ids = [CLI.label_to_id[x[1]] for x in pred_parsed.character_list]
                gold_label_ids = [CLI.label_to_id[x[1]] for x in gold_parsed.character_list]
                pred_mismatch = 0
            else:
                assert convert.seq2_type in CLI.seq2_regex, f"Unsupported convert option: {convert}"
                pred_label_ids, pred_mismatch = values_to_labels(CLI.seq2_regex[convert.seq2_type], pred_units)
                gold_label_ids, gold_mismatch = values_to_labels(CLI.seq2_regex[convert.seq2_type], gold_units)
                assert gold_mismatch == 0, f"gold_mismatch != 0: gold_mismatch={gold_mismatch}"
        else:
            assert convert.seq2_type in CLI.seq2_regex, f"Unsupported convert option: {convert}"
            pred_label_id, pred_mismatch = value_to_label(CLI.seq2_regex[convert.seq2_type], pred_units[0])
            gold_label_id, gold_mismatch = value_to_label(CLI.seq2_regex[convert.seq2_type], gold_units[0])
            assert gold_mismatch == 0, f"gold_mismatch != 0: gold_mismatch={gold_mismatch}"
            pred_label_ids = [pred_label_id]
            gold_label_ids = [gold_label_id]

        pred_tensor = torch.tensor(pred_label_ids)
        gold_tensor = torch.tensor(gold_label_ids)
        assert pred_tensor.shape == gold_tensor.shape, f"Shape mismatch: {pred_tensor.shape} != {gold_tensor.shape}"
        return pred_tensor, gold_tensor, pred_mismatch

    @staticmethod
    @main.command()
    def evaluate(
            # env
            project: str = typer.Option(default="DeepKNLU"),
            output_home: str = typer.Option(default="output"),
            logging_file: str = typer.Option(default="logging.out"),
            debugging: bool = typer.Option(default=False),
            verbose: int = typer.Option(default=1),
            # data
            input_inter: int = typer.Option(default=500000),
            refer_file_name: str = typer.Option(default="data/klue-ner/klue-ner-v1.1_dev-s2s=S1m.tsv"),
            input_file_name: str = typer.Option(default="output/klue-ner=GBST-KEByT5-Base=S1m=B4/klue-ner-v1.1_dev-s2s=S1m-last.out"),
            output_file_name: str = typer.Option(default="output/klue-ner=GBST-KEByT5-Base=S1m=B4/klue-ner-v1.1_dev-s2s=S1m-last-eval.json"),
            # convert
            s2s_type: str = typer.Option(default="S1m"),
            # evaluate
            skip_longer: bool = typer.Option(default=True),
            skip_shorter: bool = typer.Option(default=True),
    ):
        env = ProjectEnv(
            project=project,
            debugging=debugging,
            output_home=output_home,
            logging_file=logging_file,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_36 if debugging else LoggingFormat.CHECK_24,
        )
        refer_opt = InputOption(
            file=FileOption(
                name=refer_file_name,
                mode="r",
                strict=True,
            ),
        )
        input_opt = InputOption(
            inter=input_inter,
            file=FileOption(
                name=input_file_name,
                mode="r",
                strict=True,
            ),
        )
        output_opt = OutputOption(
            file=FileOption(
                name=output_file_name,
                mode="w",
                strict=True,
            ),
        )
        convert_opt = CLI.ConvertOption(
            s2s_type=s2s_type,
        )
        evaluate_opt = CLI.EvaluateOption(
            skip_longer=skip_longer,
            skip_shorter=skip_shorter,
        )
        args = CLI.EvaluateArguments(
            env=env,
            input=input_opt,
            refer=refer_opt,
            output=output_opt,
            convert=convert_opt,
            evaluate=evaluate_opt,
        )
        tqdm = mute_tqdm_cls()
        assert args.refer.file, "refer.file is required"
        assert args.input.file, "input.file is required"
        assert args.output.file, "output.file is required"

        with (
            JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}",
                     rt=1, rb=1, mb=1, rc='=', verbose=verbose > 0, args=args if debugging or verbose > 1 else None),
            FileStreamer(args.refer.file) as refer_file,
            FileStreamer(args.input.file) as input_file,
            FileStreamer(args.output.file) as output_file,
        ):
            refer_inputs, refer_labels = [], []
            for refer_item in refer_file:
                refer_seq1, refer_seq2 = refer_item.split("\t")
                if len(refer_seq1.strip()) > 0 and len(refer_seq2.strip()) > 0:
                    refer_input = CLI.strip_label_prompt(refer_seq1).splitlines()
                    refer_input = [x.split(CLI.INPUT_PROMPT)[-1].strip() for x in refer_input if CLI.INPUT_PROMPT in x]
                    refer_label = CLI.strip_label_prompt(refer_seq2).splitlines()
                    refer_input = refer_input[0] if len(refer_input) > 0 else None
                    refer_label = refer_label[0].split(CLI.EACH_SEP) if len(refer_label) > 0 else None
                    refer_inputs.append(refer_input)
                    refer_labels.append(refer_label)
            input_labels = []
            for input_label in input_file:
                input_label = CLI.strip_label_prompt(input_label).splitlines()
                input_label = input_label[0].split(CLI.EACH_SEP) if len(input_label) > 0 else None
                input_labels.append(input_label)
            logger.info(f"Load {len(refer_labels)} referring items from [{refer_file.opt}]")
            logger.info(f"Load {len(input_labels)} predicted items from [{input_file.opt}]")
            assert len(refer_inputs) == len(refer_labels), f"Length of refer_inputs and refer_labels are different: {len(refer_inputs)} != {len(refer_labels)}"
            assert len(input_labels) == len(refer_labels), f"Length of input_labels and refer_labels are different: {len(input_labels)} != {len(refer_labels)}"
            progress, interval = (
                tqdm(zip(input_labels, refer_labels, refer_inputs), total=len(input_labels), unit="item", pre="*", desc="evaluating"),
                args.input.inter,
            )

            gold_label_ids, pred_label_ids = [], []
            num_mismatched, num_skipped = 0, 0
            num_shorter, num_longer = 0, 0
            num_evaluated = 0
            for i, (pred_units, gold_units, user_input) in enumerate(progress):
                if i > 0 and i % interval == 0:
                    logger.info(progress)
                if len(pred_units) < len(gold_units):
                    num_shorter += 1
                    if args.evaluate.skip_shorter:
                        num_skipped += 1
                        continue
                if len(pred_units) > len(gold_units):
                    num_longer += 1
                    if args.evaluate.skip_longer:
                        num_skipped += 1
                        continue
                if debugging:
                    logger.info(f"-- pred_units({len(pred_units)}): {pred_units}")
                    logger.info(f"-- gold_units({len(gold_units)}): {gold_units}")

                pred_tensor, gold_tensor, pred_mismatch = CLI.units_to_label_ids(pred_units, gold_units, user_input, args.convert)
                if pred_tensor is None or gold_tensor is None:
                    num_skipped += 1
                    continue
                num_mismatched += pred_mismatch
                if debugging:
                    logger.info(f"-> pred_tensor({'x'.join(map(str, pred_tensor.shape))}): {pred_tensor.tolist()}")
                    logger.info(f"-> gold_tensor({'x'.join(map(str, gold_tensor.shape))}): {gold_tensor.tolist()}")
                    logger.info("")
                assert gold_tensor.shape == pred_tensor.shape, f"gold_tensor.shape != pred_tensor.shape: {gold_tensor.shape} != {pred_tensor.shape}"
                gold_label_ids.extend(gold_tensor.tolist())
                pred_label_ids.extend(pred_tensor.tolist())
                num_evaluated += 1
            logger.info(progress)
            assert len(gold_label_ids) == len(pred_label_ids), f"Length of gold_label_ids and pred_label_ids are different: {len(gold_label_ids)} != {len(pred_label_ids)}"

            if debugging:
                res1 = classification_report(gold_label_ids, pred_label_ids, labels=CLI.label_ids, target_names=CLI.NER_LABELS, digits=4, zero_division=1)
                logger.info(hr(c='-'))
                for line in res1.splitlines():
                    logger.info(line)

            NER_Char_MacroF1.reset()
            NER_Entity_MacroF1.reset()
            NER_Char_MacroF1.update(pred_label_ids, gold_label_ids, CLI.NER_LABELS)
            NER_Entity_MacroF1.update(pred_label_ids, gold_label_ids, CLI.NER_LABELS)

            res = EvaluateResult(
                s2s_type=args.convert.s2s_type,
                file_answer=str(refer_file.opt),
                file_predict=str(input_file.opt),
                cate_predict=input_file.path.parent.name,
                post_predict=input_file.path.stem.split("-")[-1],
                num_answer=len(refer_labels),
                num_predict=len(input_labels),
                num_evaluate=num_evaluated,
                num_mismatched=num_mismatched,
                num_skipped=num_skipped,
                num_shorter=num_shorter,
                num_longer=num_longer,
                metric_F1c=NER_Char_MacroF1.compute(),
                metric_F1e=NER_Entity_MacroF1.compute(),
            )
            output_file.fp.write(res.to_json(indent=2))
            logger.info(f"  -> s2s_type : {res.s2s_type}")
            logger.info(f"  -> num      : eval={res.num_evaluate}, miss={res.num_mismatched}, skip={res.num_skipped}, long={res.num_longer}, short={res.num_shorter}")
            logger.info(f"  -> metric   : F1c={res.metric_F1c:.2f}, F1e={res.metric_F1e:.2f}")
            logger.info(f"Save for {num_evaluated} evaluated items to [{output_file.opt}]")


if __name__ == "__main__":
    CLI.main()
