import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, ClassVar

import torch
from dataclasses_json import DataClassJsonMixin
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from chrisbase.io import make_parent_dir, files, merge_dicts
from nlpbook.arguments import TesterArguments
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import CharSpan
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy, BatchEncoding

logger = logging.getLogger("nlpbook")

NER_CLS_TOKEN = "[CLS]"
NER_SEP_TOKEN = "[SEP]"
NER_PAD_TOKEN = "[PAD]"
NER_MASK_TOKEN = "[MASK]"
NER_PAD_ID = 2


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

    def to_offset_lable_dict(self) -> dict[int, str]:
        offset_list = [(self.offset[0], f"B-{self.label}")]
        for i in range(self.offset[0] + 1, self.offset[1]):
            offset_list.append((i, f"I-{self.label}"))
        return dict(offset_list)


@dataclass
class NERExampleForKLUE(DataClassJsonMixin):
    origin: str = field(default_factory=str)
    entity_list: list[EntityInText] = field(default_factory=list)
    character_list: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class NERFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class NERCorpus:
    def __init__(self, args: TesterArguments):
        self.args = args

    def get_examples(self, data_path: Path) -> List[NERExampleForKLUE]:
        examples = []
        with data_path.open(encoding="utf-8") as inp:
            for line in inp.readlines():
                examples.append(NERExampleForKLUE.from_json(line))
        logger.info(f"Loaded {len(examples)} {examples[0].__class__.__name__} from {data_path}")
        return examples

    def get_labels(self):
        label_map_path = make_parent_dir(self.args.output.dir_path / "label_map.txt")
        if not label_map_path.exists():
            ner_tags = []
            train_data_path: Path = self.args.data.home / self.args.data.name / self.args.data.files.train
            with train_data_path.open(encoding="utf-8") as inp:
                for line in inp.readlines():
                    for x in NERExampleForKLUE.from_json(line).entity_list:
                        if x.label not in ner_tags:
                            ner_tags.append(x.label)
            b_tags = [f"B-{ner_tag}" for ner_tag in ner_tags]
            i_tags = [f"I-{ner_tag}" for ner_tag in ner_tags]
            labels = [NER_CLS_TOKEN, NER_SEP_TOKEN, NER_PAD_TOKEN, NER_MASK_TOKEN, "O"] + b_tags + i_tags
            with label_map_path.open("w", encoding="utf-8") as f:
                f.writelines([x + "\n" for x in labels])
        else:
            labels = [label.strip() for label in label_map_path.open(encoding="utf-8").readlines()]
        logger.info(f"Loaded {len(labels)} labels from {label_map_path}")
        return labels

    @property
    def num_labels(self):
        return len(self.get_labels())


def _decide_span_label(span: CharSpan, offset_to_label: dict[int, str]):
    for x in [offset_to_label[i] for i in range(span.start, span.end)]:
        if x.startswith("B-") or x.startswith("I-"):
            return x
    return "O"


def _convert_examples_to_ner_features(
        examples: List[NERExampleForKLUE],
        tokenizer: PreTrainedTokenizerFast,
        args: TesterArguments,
        label_list: List[str],
        cls_token_at_end: Optional[bool] = False,
        num_show_example: int = 3,
):
    """
    `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    """
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}

    features: list[NERFeatures] = []
    for example in examples:
        example: NERExampleForKLUE = example
        offset_to_label: dict[int, str] = {i: y for i, (_, y) in enumerate(example.character_list)}
        inputs: BatchEncoding = tokenizer.encode_plus(example.origin,
                                                      max_length=args.model.max_seq_length,
                                                      truncation=TruncationStrategy.LONGEST_FIRST,
                                                      padding=PaddingStrategy.MAX_LENGTH)
        input_tokens: List[str] = inputs.tokens()
        # out_hr()
        # print(f"offset_to_label        = {offset_to_label}")
        # out_hr()
        # print(f"input_tokens           = {inputs.tokens()}")
        # out_hr()
        # for key in inputs.keys():
        #     print(f"inputs[{key:14s}] = {inputs[key]}")
        # out_hr()

        label_list: list[str] = []
        for i in range(args.model.max_seq_length):
            token = input_tokens[i]
            token_span: CharSpan = inputs.token_to_chars(i)
            if token_span:
                token_label = _decide_span_label(token_span, offset_to_label)
                label_list.append(token_label)
                # token_str = example.origin[token_span.start:token_span.end]
                # print('\t'.join(map(str, [i, token, token_span, token_str, token_label])))
            else:
                label_list.append(token)
                # print('\t'.join(map(str, [i, token, token_span])))
        label_ids: list[int] = [label_to_id[label] for label in label_list]
        features.append(NERFeatures(**inputs, label_ids=label_ids))
        # print(f"label_list             = {label_list}")
        # out_hr()
        # print(f"label_ids              = {label_ids}")
        # print(f"features               = {features[-1]}")

    for i, example in enumerate(examples[:num_show_example]):
        logger.info("  === [Example %d] ===" % (i + 1))
        logger.info("  = sentence : %s" % example.origin)
        logger.info("  = entities : %s" % example.entity_list)
        logger.info("  = tokens   : %s" % (" ".join(tokenizer.convert_ids_to_tokens(features[i].input_ids))))
        logger.info("  = labels   : %s" % (" ".join([id_to_label[label_id] for label_id in features[i].label_ids])))
        logger.info("  = features : %s" % features[i])
        logger.info("  === ")

    return features


class NERDataset(Dataset):
    def __init__(self, split: str, args: TesterArguments, tokenizer: PreTrainedTokenizerFast, corpus: NERCorpus):
        assert corpus, "corpus is not valid"
        self.corpus = corpus

        assert args.data.home, f"No data_home: {args.data.home}"
        assert args.data.name, f"No data_name: {args.data.name}"
        data_file_dict: dict = args.data.files.to_dict()
        assert split in data_file_dict, f"No '{split}' split in data_file: should be one of {list(data_file_dict.keys())}"
        assert data_file_dict[split], f"No data_file for '{split}' split: {args.data.files}"
        text_data_path: Path = Path(args.data.home) / args.data.name / data_file_dict[split]
        cache_data_path = text_data_path \
            .with_stem(text_data_path.stem + f"-by-{tokenizer.__class__.__name__}-with-{args.model.max_seq_length}") \
            .with_suffix(".cache")
        cache_lock_path = cache_data_path.with_suffix(".lock")

        with FileLock(cache_lock_path):
            if os.path.exists(cache_data_path) and args.data.caching:
                start = time.time()
                self.features = torch.load(cache_data_path)
                logger.info(f"Loading features from cached file at {cache_data_path} [took {time.time() - start:.3f} s]")
            else:
                assert text_data_path.exists() and text_data_path.is_file(), f"No data_text_path: {text_data_path}"
                logger.info(f"Creating features from dataset file at {text_data_path}")
                examples = self.corpus.get_examples(text_data_path)
                self.features = _convert_examples_to_ner_features(examples, tokenizer, args, label_list=self.corpus.get_labels())
                start = time.time()
                torch.save(self.features, cache_data_path)
                logger.info(f"Saving features into cached file at {cache_data_path} [took {time.time() - start:.3f} s]")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.corpus.get_labels()


def _parse_tagged(origin: str, tagged: str, debug: bool = False) -> Optional[NERExampleForKLUE]:
    entity_list: list[EntityInText] = []
    if debug:
        print(f"* origin: {origin}")
        print(f"  tagged: {tagged}")
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
            print(f"  = {entity} -> {extracted}")
            # print(f"    {offset_labels}")
    if debug:
        print(f"  --------------------")
    character_list = [(origin[i], offset_labels[i]) for i in range(len(origin))]
    if restored != origin:
        no_problem = False
    return NERExampleForKLUE(origin, entity_list, character_list) if no_problem else None


def convert_kmou_format(infile: str | Path, outfile: str | Path, debug: bool = False):
    with Path(infile).open(encoding="utf-8") as inp, Path(outfile).open("w", encoding="utf-8") as out:
        for line in inp.readlines():
            origin, tagged = line.strip().split("\u241E")
            parsed: Optional[NERExampleForKLUE] = _parse_tagged(origin, tagged, debug=debug)
            if parsed:
                out.write(parsed.to_json(ensure_ascii=False) + "\n")


def convert_klue_format(infile: str | Path, outfile: str | Path, debug: bool = False):
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
            parsed: Optional[NERExampleForKLUE] = _parse_tagged(origin, tagged, debug=debug)
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


if __name__ == "__main__":
    for path in files("data/kmou-ner-full/*.txt"):
        print(f"[FILE]: {path}")
        convert_kmou_format(path, path.with_suffix(".jsonl"), debug=True)

    # for path in files("data/klue-ner/*.tsv"):
    #     print(f"[FILE]: {path}")
    #     convert_klue_format(path, path.with_suffix(".jsonl"), debug=True)
