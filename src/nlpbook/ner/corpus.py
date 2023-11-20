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
from torch.utils.data.dataset import Dataset
from transformers import CharSpan
from transformers import PreTrainedTokenizerFast, BatchEncoding
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from chrisbase.data import AppTyper, ProjectEnv, InputOption, FileOption, IOArguments, OutputOption, JobTimer, FileStreamer, OptionData
from chrisbase.io import hr, LoggingFormat, file_size, cwd
from chrisbase.io import make_parent_dir, merge_dicts
from chrisbase.util import mute_tqdm_cls, LF, HT, NO
from chrisbase.util import to_dataframe
from nlpbook.arguments import MLArguments, TrainerArguments

logger = logging.getLogger(__name__)


@dataclass
class EntityInText(DataClassJsonMixin):
    pattern: ClassVar[re.Pattern] = re.compile('<([^<>]+?):([A-Z]{2,3})>')
    text: str
    label: str
    offset: tuple[int, int]

    @staticmethod
    def from_match(m: re.Match, s: str) -> tuple["EntityInText", str]:
        x = m.group(1)
        y = m.group(2)
        z = (m.start(), m.start() + len(x))
        e = EntityInText(text=x, label=y, offset=z)
        s = s[:m.start()] + m.group(1) + s[m.end():]
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
                             data_path: str | Path = "klue-ner/klue-ner-v1.1_dev.jsonl",
                             label_path: str | Path = "klue-ner/label_map.txt") -> List[str]:
        label_path = make_parent_dir(label_path).absolute()
        if not label_path.exists():
            data_path0 = data_path
            data_path = Path(data_path).absolute()
            data_path = data_path if data_path.exists() else None
            assert data_path, f"No data_path: {data_path0}"
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
        else:
            labels = label_path.read_text().splitlines()
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


class CLI:
    main = AppTyper()
    task = "Named Entity Recognition"
    LINE_SEP = "<LF>"
    EACH_SEP = "▁"
    MAIN_PROMPT = f"{task} on Sentence: "
    EACH_PROMPT = f"{task} on Character: "
    cwdcwd = cwd()
    label_names = NERCorpus.get_labels_from_data(data_path="klue-ner/klue-ner-v1.1_dev.jsonl",
                                                 label_path="klue-ner/label_map.txt")
    label_ids = [i for i, _ in enumerate(label_names)]
    label_to_id = {label: i for i, label in enumerate(label_names)}
    id_to_label = {i: label for i, label in enumerate(label_names)}

    @classmethod
    def strip_label_prompt(cls, x: str):
        x = x.replace(cls.LINE_SEP, LF)
        x = x.replace(cls.MAIN_PROMPT, NO)
        x = x.replace(cls.EACH_PROMPT, NO)
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
                        print(CLI.MAIN_PROMPT + main_label, file=s)
                    else:
                        print(CLI.MAIN_PROMPT + CLI.EACH_SEP.join(sub_labels), file=s)
                        print(f"Label Count: {len(sub_labels)}", file=s)
                    seq2 = [CLI.to_str(s)]
            else:
                if len(sub_labels) != len(seq1):
                    return []
                seq2 = [CLI.EACH_PROMPT + label for label in sub_labels]

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
    def repr_to_labels(units: List[str], convert: "CLI.ConvertOption"):
        print(f"units: {units}")
        print(f"convert: {convert}")
        exit(1)
        heads = [-1] * len(units)
        types = [-1] * len(units)
        assert convert.seq2_type in CLI.seq2_regex, f"Unsupported convert option: {convert}"
        regex = CLI.seq2_regex[convert.seq2_type]
        num_mismatch = 0
        for i, x in enumerate(units):
            m = regex['pattern'].search(x)
            if m:
                dep = m.group(regex['dep'])
                head = m.group(regex['head'])
                dep_id = CLI.label_to_id.get(dep, 0) if dep else 0
                head_id = int(head)
                heads[i] = head_id
                types[i] = dep_id
            else:
                logger.warning(f"Not found: pattern={regex['pattern']}, string={x}")
                num_mismatch += 1
                heads[i] = 0
                types[i] = 0
        result = DPResult(torch.tensor(heads), torch.tensor(types))
        assert result.heads.shape == result.types.shape, f"result.heads.shape != result.types.shape: {result.heads.shape} != {result.types.shape}"
        return result, num_mismatch

    # nlpbook.ner.corpus evaluate --input-file-name output/klue-ner=GBST-KEByT5-Base=S0a=B4/klue-ner-v1.1_dev-s2s=S0a-1969.out --refer-file-name data/klue-ner/klue-ner-v1.1_dev-s2s=S0a.tsv --output-file-name output/klue-ner=GBST-KEByT5-Base=S0a=B4/klue-ner-v1.1_dev-s2s=S0a-1969-eval.json --s2s-type S0a --verbose 0
    @staticmethod
    @main.command()
    def evaluate(
            # env
            project: str = typer.Option(default="DeepKNLU"),
            output_home: str = typer.Option(default="output"),
            logging_file: str = typer.Option(default="logging.out"),
            debugging: bool = typer.Option(default=True),
            verbose: int = typer.Option(default=1),
            # data
            input_inter: int = typer.Option(default=50000),
            refer_file_name: str = typer.Option(default="data/klue-ner/klue-ner-v1.1_dev-s2s=S0a.tsv"),
            input_file_name: str = typer.Option(default="output/klue-ner=GBST-KEByT5-Base=S0a=B4/klue-ner-v1.1_dev-s2s=S0a-1969.out"),
            output_file_name: str = typer.Option(default="output/klue-ner=GBST-KEByT5-Base=S0a=B4/klue-ner-v1.1_dev-s2s=S0a-1969-eval.json"),
            # convert
            s2s_type: str = typer.Option(default="S0a"),
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
            refer_items = [x.strip() for x in [x.split("\t")[1] for x in refer_file] if len(x.strip()) > 0]
            input_items = [x.strip() for x in [x for x in input_file] if len(x.strip()) > 0]
            logger.info(f"Load {len(refer_items)}  labelled items from [{refer_file.opt}]")
            logger.info(f"Load {len(input_items)} predicted items from [{input_file.opt}]")
            assert len(input_items) == len(refer_items), f"Length of input_items and refer_items are different: {len(input_items)} != {len(refer_items)}"
            progress, interval = (
                tqdm(zip(input_items, refer_items), total=len(input_items), unit="item", pre="*", desc="evaluating"),
                args.input.inter,
            )

            golds, preds = [], []
            num_mismatched, num_skipped = 0, 0
            num_shorter, num_longer = 0, 0
            for i, (a, b) in enumerate(progress):
                if i > 0 and i % interval == 0:
                    logger.info(progress)
                pred_units = CLI.strip_label_prompt(a).splitlines()[0].split(CLI.EACH_SEP)
                gold_units = CLI.strip_label_prompt(b).splitlines()[0].split(CLI.EACH_SEP)
                if len(pred_units) < len(gold_units):
                    num_shorter += 1
                    if args.evaluate.skip_shorter:
                        num_skipped += 1
                        continue
                    else:
                        logger.warning(f"[{i:04d}] Shorter pred_units({len(pred_units)}): {pred_units}")
                        logger.warning(f"[{i:04d}]         gold_units({len(gold_units)}): {gold_units}")
                        pred_units = pred_units + (
                                [pred_units[-1]] * (len(gold_units) - len(pred_units))
                        )
                        logger.warning(f"[{i:04d}]      -> pred_units({len(pred_units)}): {pred_units}")
                if len(pred_units) > len(gold_units):
                    num_longer += 1
                    if args.evaluate.skip_longer:
                        num_skipped += 1
                        continue
                    else:
                        logger.warning(f"[{i:04d}]  Longer pred_units({len(pred_units)}): {pred_units}")
                        logger.warning(f"[{i:04d}]         gold_units({len(gold_units)}): {gold_units}")
                        pred_units = pred_units[:len(gold_units)]
                        logger.warning(f"[{i:04d}]      -> pred_units({len(pred_units)}): {pred_units}")
                assert len(pred_units) == len(gold_units), f"Length of pred_units and gold_units are different: {len(pred_units)} != {len(gold_units)}"
                if debugging:
                    logger.info(f"-- pred_units({len(pred_units)}): {pred_units}")
                    logger.info(f"-- gold_units({len(gold_units)}): {gold_units}")

                pred_units, pred_mismatch = CLI.repr_to_labels(pred_units, args.convert)
                exit(1)
                gold_units, gold_mismatch = CLI.repr_to_labels(gold_units, args.convert)
                assert gold_mismatch == 0, f"gold_mismatch != 0: gold_mismatch={gold_mismatch}"

                exit(1)


if __name__ == "__main__":
    CLI.main()
