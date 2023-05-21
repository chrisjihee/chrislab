import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, ClassVar, Dict

import torch
from dataclasses_json import DataClassJsonMixin
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from chrisbase.io import make_parent_dir, files, merge_dicts, out_hr
from nlpbook.arguments import TesterArguments
from transformers import PreTrainedTokenizerFast, BatchEncoding, CharSpan
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

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

    def to_offset_lable_dict(self) -> Dict[int, str]:
        offset_list = [(self.offset[0], f"B-{self.label}")]
        for i in range(self.offset[0] + 1, self.offset[1]):
            offset_list.append((i, f"I-{self.label}"))
        return dict(offset_list)


@dataclass
class NERRawExample(DataClassJsonMixin):
    origin: str = field(default_factory=str)
    entity_list: List[EntityInText] = field(default_factory=list)
    character_list: List[tuple[str, str]] = field(default_factory=list)

    def get_offset_label_dict(self):
        return {i: y for i, (_, y) in enumerate(self.character_list)}


@dataclass
class NEREncodedExample:
    idx: int
    raw: NERRawExample
    encoded: BatchEncoding
    label_ids: Optional[List[int]] = None


def encoded_examples_to_batch(examples: List[NEREncodedExample]) -> Dict[str, torch.Tensor | List[int]]:
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
    batch["example_ids"] = [ex.idx for ex in examples]
    return batch


class NERCorpus:
    def __init__(self, args: TesterArguments):
        self.args = args

    def read_raw_examples(self, data_path: Path) -> List[NERRawExample]:
        examples = []
        with data_path.open(encoding="utf-8") as inp:
            for line in inp.readlines():
                examples.append(NERRawExample.from_json(line))
        logger.info(f"Loaded {len(examples)} examples from {data_path}")
        return examples

    def get_labels(self) -> List[str]:
        label_map_path = make_parent_dir(self.args.output.dir_path / "label_map.txt")
        if not label_map_path.exists():
            ner_tags = []
            train_data_path = self.args.data.home / self.args.data.name / self.args.data.files.train
            logger.info(f"Extracting labels from {train_data_path}")
            with train_data_path.open(encoding="utf-8") as inp:
                for line in inp.readlines():
                    for x in NERRawExample.from_json(line).entity_list:
                        if x.label not in ner_tags:
                            ner_tags.append(x.label)
            b_tags = [f"B-{ner_tag}" for ner_tag in ner_tags]
            i_tags = [f"I-{ner_tag}" for ner_tag in ner_tags]
            labels = ["O"] + b_tags + i_tags  # TODO: Opt1: No special label
            # labels = [NER_CLS_TOKEN, NER_SEP_TOKEN, NER_PAD_TOKEN, NER_MASK_TOKEN, "O"] + b_tags + i_tags  # TODO: Opt2: Use special labels
            logger.info(f"Saved {len(labels)} labels to {label_map_path}")
            with label_map_path.open("w", encoding="utf-8") as f:
                f.writelines([x + "\n" for x in labels])
        else:
            labels = label_map_path.read_text(encoding="utf-8").splitlines()
            logger.info(f"Loaded {len(labels)} labels from {label_map_path}")
        return labels

    @property
    def num_labels(self):
        return len(self.get_labels())


def _decide_span_label(span: CharSpan, offset_to_label: Dict[int, str]):
    for x in [offset_to_label[i] for i in range(span.start, span.end)]:
        if x.startswith("B-") or x.startswith("I-"):
            return x
    return "O"


def _convert_to_encoded_examples(
        raw_examples: List[NERRawExample],
        tokenizer: PreTrainedTokenizerFast,
        args: TesterArguments,
        label_list: List[str],
        cls_token_at_end: Optional[bool] = False,
        num_show_example: int = 3,
) -> List[NEREncodedExample]:
    """
    `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    """
    label_to_id: Dict[str, int] = {label: i for i, label in enumerate(label_list)}
    id_to_label: Dict[int, str] = {i: label for i, label in enumerate(label_list)}
    if args.env.off_debugging:
        print(f"label_to_id = {label_to_id}")
        print(f"id_to_label = {id_to_label}")

    encoded_examples: List[NEREncodedExample] = []
    for idx, raw_example in enumerate(raw_examples):
        raw_example: NERRawExample = raw_example
        offset_to_label: Dict[int, str] = raw_example.get_offset_label_dict()
        if args.env.off_tracing:
            out_hr()
            print(f"offset_to_label = {offset_to_label}")
        encoded: BatchEncoding = tokenizer.encode_plus(raw_example.origin,
                                                       max_length=args.model.max_seq_length,
                                                       truncation=TruncationStrategy.LONGEST_FIRST,
                                                       padding=PaddingStrategy.MAX_LENGTH)
        encoded_tokens: List[str] = encoded.tokens()
        if args.env.off_debugging:
            out_hr()
            print(f"encoded.tokens()           = {encoded.tokens()}")
            for key in encoded.keys():
                print(f"encoded[{key:14s}]    = {encoded[key]}")

        if args.env.off_tracing:
            out_hr()
        label_list: List[str] = []
        for token_id in range(args.model.max_seq_length):
            token_repr: str = encoded_tokens[token_id]
            token_span: CharSpan = encoded.token_to_chars(token_id)
            if token_span:
                token_label = _decide_span_label(token_span, offset_to_label)
                label_list.append(token_label)
                if args.env.off_tracing:
                    token_sstr = raw_example.origin[token_span.start:token_span.end]
                    print('\t'.join(map(str, [token_id, token_repr, token_span, token_sstr, token_label])))
            else:
                label_list.append('O')  # TODO: Opt1: No special label
                # label_list.append(token_repr)  # TODO: Opt2: Use special labels
                if args.env.off_tracing:
                    print('\t'.join(map(str, [token_id, token_repr, token_span])))
        label_ids: List[int] = [label_to_id[label] for label in label_list]
        encoded_example = NEREncodedExample(idx=idx, raw=raw_example, encoded=encoded, label_ids=label_ids)
        encoded_examples.append(encoded_example)
        if args.env.off_debugging:
            out_hr()
            print(f"label_list                = {label_list}")
            print(f"label_ids                 = {label_ids}")
            out_hr()
            print(f"encoded_example.idx       = {encoded_example.idx}")
            print(f"encoded_example.raw       = {encoded_example.raw}")
            print(f"encoded_example.encoded   = {encoded_example.encoded}")
            print(f"encoded_example.label_ids = {encoded_example.label_ids}")

    if args.env.off_debugging:
        out_hr()
    for encoded_example in encoded_examples[:num_show_example]:
        logger.info("  === [Example %d] ===" % encoded_example.idx)
        logger.info("  = sentence   : %s" % encoded_example.raw.origin)
        logger.info("  = characters : %s" % " | ".join(f"{x}/{y}" for x, y in encoded_example.raw.character_list))
        logger.info("  = tokens     : %s" % " ".join(encoded_example.encoded.tokens()))
        logger.info("  = labels     : %s" % " ".join([id_to_label[x] for x in encoded_example.label_ids]))
        logger.info("  === ")

    return encoded_examples


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
            .with_stem(text_data_path.stem + f"-by-{tokenizer.__class__.__name__}-as-{args.model.max_seq_length}toks") \
            .with_suffix(".cache")
        cache_lock_path = cache_data_path.with_suffix(".lock")

        with FileLock(cache_lock_path):
            if os.path.exists(cache_data_path) and args.data.caching:
                start = time.time()
                self.features: List[NEREncodedExample] = torch.load(cache_data_path)
                logger.info(f"Loading features from cached file at {cache_data_path} [took {time.time() - start:.3f} s]")
            else:
                assert text_data_path.exists() and text_data_path.is_file(), f"No data_text_path: {text_data_path}"
                logger.info(f"Creating features from dataset file at {text_data_path}")
                examples: List[NERRawExample] = self.corpus.read_raw_examples(text_data_path)
                self.features: List[NEREncodedExample] = _convert_to_encoded_examples(examples, tokenizer, args,
                                                                                      label_list=self.corpus.get_labels())
                if args.data.caching:
                    start = time.time()
                    torch.save(self.features, cache_data_path)
                    logger.info(f"Saving features into cached file at {cache_data_path} [took {time.time() - start:.3f} s]")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i) -> NEREncodedExample:
        return self.features[i]

    def get_labels(self) -> List[str]:
        return self.corpus.get_labels()


def parse_tagged(origin: str, tagged: str, debug: bool = False) -> Optional[NERRawExample]:
    entity_list: List[EntityInText] = []
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
    return NERRawExample(origin, entity_list, character_list) if no_problem else None


def convert_kmou_format(infile: str | Path, outfile: str | Path, debug: bool = False):
    with Path(infile).open(encoding="utf-8") as inp, Path(outfile).open("w", encoding="utf-8") as out:
        for line in inp.readlines():
            origin, tagged = line.strip().split("\u241E")
            parsed: Optional[NERRawExample] = parse_tagged(origin, tagged, debug=debug)
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
            parsed: Optional[NERRawExample] = parse_tagged(origin, tagged, debug=debug)
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
