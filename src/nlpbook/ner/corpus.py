import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, ClassVar, Dict

import torch
import typer
from dataclasses_json import DataClassJsonMixin
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding, CharSpan
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
from chrisbase.data import AppTyper, ProjectEnv, InputOption, FileOption, IOArguments, OutputOption, JobTimer, FileStreamer, OptionData, ResultData

from chrisbase.io import make_parent_dir, files, merge_dicts, hr, LoggingFormat
from nlpbook.arguments import MLArguments

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

from chrisbase.data import AppTyper, ProjectEnv, InputOption, FileOption, IOArguments, OutputOption, JobTimer, FileStreamer, OptionData, ResultData
from chrisbase.io import hr, LoggingFormat
from chrisbase.util import mute_tqdm_cls, LF, HT, NO
from chrisbase.util import to_dataframe
from nlpbook.arguments import TesterArguments, TrainerArguments
from nlpbook.metrics import DPResult, DP_UAS_MacroF1, DP_LAS_MacroF1, DP_UAS_MicroF1, DP_LAS_MicroF1

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

    def get_labels(self) -> List[str]:
        label_map_path = make_parent_dir(self.args.env.output_home / "label_map.txt")
        if not label_map_path.exists():
            ner_tags = []
            train_data_path = self.args.data.home / self.args.data.name / self.args.data.files.train
            logger.info(f"Extracting labels from {train_data_path}")
            with train_data_path.open(encoding="utf-8") as inp:
                for line in inp.readlines():
                    for x in NERParsedExample.from_json(line).entity_list:
                        if x.label not in ner_tags:
                            ner_tags.append(x.label)
            b_tags = [f"B-{ner_tag}" for ner_tag in ner_tags]
            i_tags = [f"I-{ner_tag}" for ner_tag in ner_tags]
            labels = ["O"] + b_tags + i_tags
            logger.info(f"Saved {len(labels)} labels to {label_map_path}")
            with label_map_path.open("w", encoding="utf-8") as f:
                f.writelines([x + "\n" for x in labels])
        else:
            labels = label_map_path.read_text(encoding="utf-8").splitlines()
        return labels

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

    @classmethod
    def convert_to_seq2seq_format_v1(cls, infile: str | Path, outfile1: str | Path, outfile2: str | Path = None, debug: bool = False):
        # TODO:
        #  1) 문장 -> 글자+태그
        with Path(infile).open(encoding="utf-8") as inp:
            out1 = Path(outfile1).open("w", encoding="utf-8") if outfile1 else None
            out2 = Path(outfile2).open("w", encoding="utf-8") if outfile2 else None
            for line in inp.readlines():
                example = NERParsedExample.from_json(line)
                seq1 = example.origin
                seq2 = ' '.join([f"{c}/{t}" for c, t in example.character_list if c != ' '])
                if out1 and out2:
                    out1.write(f"{seq1}\n")
                    out2.write(f"{seq2}\n")
                elif out1:
                    out1.write(f"{seq1}\t{seq2}\n")
            if out1:
                out1.close()
            if out2:
                out2.close()

    @classmethod
    def convert_to_seq2seq_format_v2(cls, infile: str | Path, outfile1: str | Path, outfile2: str | Path = None, debug: bool = False):
        # TODO:
        #  2) 문장 -> 문장 내에 개체명을 레이블한 결과
        with Path(infile).open(encoding="utf-8") as inp:
            out1 = Path(outfile1).open("w", encoding="utf-8") if outfile1 else None
            out2 = Path(outfile2).open("w", encoding="utf-8") if outfile2 else None
            for line in inp.readlines():
                example = NERParsedExample.from_json(line)
                seq1 = example.origin
                seq2 = example.to_tagged_text(lambda e: f"<{e.text}:{e.label}>")
                if out1 and out2:
                    out1.write(f"{seq1}\n")
                    out2.write(f"{seq2}\n")
                elif out1:
                    out1.write(f"{seq1}\t{seq2}\n")
            if out1:
                out1.close()
            if out2:
                out2.close()

    @classmethod
    def convert_to_seq2seq_format_v3(cls, infile: str | Path, outfile1: str | Path, outfile2: str | Path = None, debug: bool = False):
        # TODO:
        #  3) 문장 + 오프셋 -> 해당 오프셋 글자에 대한 태그 (학습데이터 많아짐)
        with Path(infile).open(encoding="utf-8") as inp:
            out1 = Path(outfile1).open("w", encoding="utf-8") if outfile1 else None
            out2 = Path(outfile2).open("w", encoding="utf-8") if outfile2 else None
            for line in inp.readlines():
                example = NERParsedExample.from_json(line)
                s = example.origin
                n = 0
                for c, t in example.character_list:
                    n += 1
                    if c == ' ':
                        continue
                    seq1 = f"질문: 문장에서 {n}번째 글자인 <{c}>의 개체명 태그는? 문장: {s}"
                    seq2 = t
                    if out1 and out2:
                        out1.write(f"{seq1}\n")
                        out2.write(f"{seq2}\n")
                    elif out1:
                        out1.write(f"{seq1}\t{seq2}\n")
            if out1:
                out1.close()
            if out2:
                out2.close()


class CLI:
    main = AppTyper()
    task = "Named Entity Recognition"
    LINE_SEP = "<LF>"

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

        def to_seq1(self, example: NERParsedExample):
            forms = []
            for idx, (char, _) in enumerate(example.character_list, start=1):
                if char.strip():
                    forms.append(f'{idx}/{char}')

            with StringIO() as s:
                print(f"Task: {CLI.task}", file=s)
                print('', file=s)
                print(f"Input: {example.origin}", file=s)

                if self.convert.seq1_type == 'S0':
                    return [CLI.to_str(s)]

                elif self.convert.seq1_type in ('S1', 'C1'):
                    print(f"Forms: {LF.join(forms)}", file=s)
                    if self.convert.seq1_type == 'S1':
                        return [CLI.to_str(s)]
                    elif self.convert.seq1_type == 'C1':
                        return [CLI.to_str(s) + f"Target: {form}" + CLI.LINE_SEP for form in forms]
                    else:
                        raise NotImplementedError(f"Unsupported convert: {self.convert}")

                else:
                    raise NotImplementedError(f"Unsupported convert: {self.convert}")

    @staticmethod
    def to_str(s: StringIO):
        s.seek(0)
        return s.read().replace(LF, CLI.LINE_SEP)

    @staticmethod
    @main.command()
    def convert(
            # env
            project: str = typer.Option(default="DeepKNLU"),
            job_name: str = typer.Option(default="convert"),
            output_home: str = typer.Option(default="output"),
            logging_file: str = typer.Option(default="logging.out"),
            debugging: bool = typer.Option(default=False),
            verbose: int = typer.Option(default=1),
            # data
            input_inter: int = typer.Option(default=5000),
            input_file_name: str = typer.Option(default="data/klue-ner/klue-ner-v1.1_dev.tsv"),
            output_file_name: str = typer.Option(default="data/klue-ner/klue-ner-v1.1_dev-s2s.tsv"),
            # convert
            s2s_type: str = typer.Option(default="C1a"),
    ):
        env = ProjectEnv(
            project=project,
            job_name=job_name,
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
                print(f"example: {example}")
                seq1s = args.to_seq1(example)
                print(f"seq1s: {seq1s}")
                exit(1)


if __name__ == "__main__":
    CLI.main()

# if __name__ == "__main__":
#     class RunOption:
#         run1: bool = False
#         run2: bool = False
#         run3_v1: bool = True
#         run3_v2: bool = True
#         run3_v3: bool = True
#
#
#     if RunOption.run1:
#         for path in files("data/kmou-ner-full/*.txt"):
#             print(f"[FILE]: {path}")
#             NERCorpusConverter.convert_from_kmou_format(path, path.with_suffix(".jsonl"), debug=True)
#
#     if RunOption.run2:
#         for path in files("data/klue-ner/*.tsv"):
#             print(f"[FILE]: {path}")
#             NERCorpusConverter.convert_from_klue_format(path, path.with_suffix(".jsonl"), debug=True)
#
#     if RunOption.run3_v1:
#         for path in files("data/klue-ner-mini/*_dev.jsonl") + files("data/klue-ner/*_dev.jsonl"):
#             print(f"[FILE]: {path}")
#             NERCorpusConverter.convert_to_seq2seq_format_v1(path, path.with_suffix(".input.seq2seq_v1.tsv"), path.with_suffix(".answer.seq2seq_v1.tsv"), debug=True)
#         for path in files("data/klue-ner-mini/*_train.jsonl") + files("data/klue-ner/*_train.jsonl"):
#             print(f"[FILE]: {path}")
#             NERCorpusConverter.convert_to_seq2seq_format_v1(path, path.with_suffix(".seq2seq_v1.tsv"), debug=True)
#
#     if RunOption.run3_v2:
#         for path in files("data/klue-ner-mini/*_dev.jsonl") + files("data/klue-ner/*_dev.jsonl"):
#             print(f"[FILE]: {path}")
#             NERCorpusConverter.convert_to_seq2seq_format_v2(path, path.with_suffix(".input.seq2seq_v2.tsv"), path.with_suffix(".answer.seq2seq_v2.tsv"), debug=True)
#         for path in files("data/klue-ner-mini/*_train.jsonl") + files("data/klue-ner/*_train.jsonl"):
#             print(f"[FILE]: {path}")
#             NERCorpusConverter.convert_to_seq2seq_format_v2(path, path.with_suffix(".seq2seq_v2.tsv"), debug=True)
#
#     if RunOption.run3_v3:
#         for path in files("data/klue-ner-mini/*_dev.jsonl") + files("data/klue-ner/*_dev.jsonl"):
#             print(f"[FILE]: {path}")
#             NERCorpusConverter.convert_to_seq2seq_format_v3(path, path.with_suffix(".input.seq2seq_v3.tsv"), path.with_suffix(".answer.seq2seq_v3.tsv"), debug=True)
#         for path in files("data/klue-ner-mini/*_train.jsonl") + files("data/klue-ner/*_train.jsonl"):
#             print(f"[FILE]: {path}")
#             NERCorpusConverter.convert_to_seq2seq_format_v3(path, path.with_suffix(".seq2seq_v3.tsv"), debug=True)
