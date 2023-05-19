import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Iterable, Optional, ClassVar

import torch
from dataclasses_json import DataClassJsonMixin
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from chrisbase.io import make_parent_dir, files, merge_dicts
from nlpbook.arguments import TrainerArguments, TesterArguments
from transformers import BertTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy, BatchEncoding

logger = logging.getLogger("nlpbook")

# 자체 제작 NER 코퍼스 기준의 레이블 시퀀스를 만들기 위한 ID 체계
# 나 는 삼성 에 입사 했다
# O O 기관 O O O > [CLS] O O 기관 O O O [SEP] [PAD] [PAD] ...
NER_CLS_TOKEN = "[CLS]"
NER_SEP_TOKEN = "[SEP]"
NER_PAD_TOKEN = "[PAD]"
NER_MASK_TOKEN = "[MASK]"
NER_PAD_ID = 2


@dataclass
class NERExample:
    text: str
    label: Optional[str] = None


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
    def __init__(
            self,
            args: TrainerArguments | TesterArguments,
    ):
        self.args = args

    def get_examples(self, data_path: Path) -> List[NERExampleForKLUE] | List[NERExample]:
        examples = []
        if data_path.suffix.lower() == ".jsonl":
            with data_path.open(encoding="utf-8") as inp:
                for line in inp.readlines():
                    examples.append(NERExampleForKLUE.from_json(line))
        else:
            for line in open(data_path, "r", encoding="utf-8").readlines():
                text, label = line.split("\u241E")
                examples.append(NERExample(text=text, label=label))
        logger.info(f"Loaded {len(examples)} {examples[0].__class__.__name__} from {data_path}")
        return examples

    def get_labels(self):
        label_map_path = make_parent_dir(self.args.output.dir_path / "label_map.txt")
        if not label_map_path.exists():
            logger.info("processing NER tag dictionary...")
            os.makedirs(self.args.model.finetuning_home, exist_ok=True)
            ner_tags = []
            regex_ner = re.compile('<(.+?):[A-Z]{3}>')
            train_corpus_path = self.args.data.home / self.args.data.name / "train.txt"
            target_sentences = [line.split("\u241E")[1].strip()
                                for line in train_corpus_path.open("r", encoding="utf-8").readlines()]
            for target_sentence in target_sentences:
                regex_filter_res = regex_ner.finditer(target_sentence)
                for match_item in regex_filter_res:
                    ner_tag = match_item[0][-4:-1]
                    if ner_tag not in ner_tags:
                        ner_tags.append(ner_tag)
            b_tags = [f"B-{ner_tag}" for ner_tag in ner_tags]
            i_tags = [f"I-{ner_tag}" for ner_tag in ner_tags]
            labels = [NER_CLS_TOKEN, NER_SEP_TOKEN, NER_PAD_TOKEN, NER_MASK_TOKEN, "O"] + b_tags + i_tags
            with label_map_path.open("w", encoding="utf-8") as f:
                for tag in labels:
                    f.writelines(tag + "\n")
        else:
            labels = [tag.strip() for tag in open(label_map_path, "r", encoding="utf-8").readlines()]
        return labels

    @property
    def num_labels(self):
        return len(self.get_labels())


def _process_target_sentence(
        tokens: List[str],
        origin_sentence: str,
        target_sentence: str,
        max_length: int,
        label_map: dict,
        tokenizer: BertTokenizer,
        cls_token_at_end: Optional[bool] = False,
):
    """
    target_sentence = "―<효진:PER> 역의 <김환희:PER>(<14:NOH>)가 특히 인상적이었다."
    tokens = ["―", "효", "##진", "역", "##의", "김", "##환", "##희",
              "(", "14", ")", "가", "특히", "인상", "##적이", "##었다", "."]
    label_sequence = ['O', 'B-PER', 'I-PER', 'O', 'O', 'B-PER', 'I-PER', 'I-PER', 'O',
                      'B-NOH', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    """
    if "[UNK]" in tokens:
        processed_tokens = []
        basic_tokens = tokenizer.basic_tokenizer.tokenize(origin_sentence)
        for basic_token in basic_tokens:
            current_tokens = tokenizer.tokenize(basic_token)
            if "[UNK]" in current_tokens:
                # [UNK] 복원
                processed_tokens.append(basic_token)
            else:
                processed_tokens.extend(current_tokens)
    else:
        processed_tokens = tokens

    prefix_sum_of_token_start_index, sum = [0], 0
    for i, token in enumerate(processed_tokens):
        if token.startswith("##"):
            sum += len(token) - 2
        else:
            sum += len(token)
        prefix_sum_of_token_start_index.append(sum)

    regex_ner = re.compile('<(.+?):[A-Z]{3}>')  # NER Tag가 2자리 문자면 {3} -> {2}로 변경 (e.g. LOC -> LC) 인경우
    regex_filter_res = regex_ner.finditer(target_sentence.replace(" ", ""))

    list_of_ner_tag = []
    list_of_ner_text = []
    list_of_tuple_ner_start_end = []

    count_of_match = 0
    for match_item in regex_filter_res:
        ner_tag = match_item[0][-4:-1]  # <4일간:DUR> -> DUR
        ner_text = match_item[1]  # <4일간:DUR> -> 4일간
        start_index = match_item.start() - 6 * count_of_match  # delete previous '<, :, 3 words tag name, >'
        end_index = match_item.end() - 6 - 6 * count_of_match

        list_of_ner_tag.append(ner_tag)
        list_of_ner_text.append(ner_text)
        list_of_tuple_ner_start_end.append((start_index, end_index))
        count_of_match += 1

    label_sequence = []
    entity_index = 0
    is_entity_still_B = True

    for tup in zip(processed_tokens, prefix_sum_of_token_start_index):
        token, index = tup

        if entity_index < len(list_of_tuple_ner_start_end):
            start, end = list_of_tuple_ner_start_end[entity_index]

            if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                is_entity_still_B = True
                entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end) else entity_index
                start, end = list_of_tuple_ner_start_end[entity_index]

            if start <= index and index < end:  # <13일:DAT>까지 -> ('▁13', 10, 'B-DAT') ('일까지', 12, 'I-DAT') 이런 경우가 포함됨, 포함 안시키려면 토큰의 length도 계산해서 제어해야함
                entity_tag = list_of_ner_tag[entity_index]
                if is_entity_still_B is True:
                    entity_tag = 'B-' + entity_tag
                    label_sequence.append(entity_tag)
                    is_entity_still_B = False
                else:
                    entity_tag = 'I-' + entity_tag
                    label_sequence.append(entity_tag)
            else:
                is_entity_still_B = True
                entity_tag = 'O'
                label_sequence.append(entity_tag)
        else:
            entity_tag = 'O'
            label_sequence.append(entity_tag)

    # truncation
    label_sequence = label_sequence[:max_length - 2]

    # add special tokens
    if cls_token_at_end:
        label_sequence = label_sequence + [NER_CLS_TOKEN, NER_SEP_TOKEN]
    else:
        label_sequence = [NER_CLS_TOKEN] + label_sequence + [NER_SEP_TOKEN]

    # padding
    pad_length = max(max_length - len(label_sequence), 0)
    pad_sequence = [NER_PAD_TOKEN] * pad_length
    label_sequence += pad_sequence

    # encoding
    label_ids = [label_map[label] for label in label_sequence]
    return label_ids


def _decide_span_label(labels: Iterable[str]):
    for label in labels:
        if label.startswith("B-") or label.startswith("I-"):
            return label
    return "O"


def _convert_examples_to_ner_features(
        examples: List[NERExampleForKLUE],  # |List[NERExample]
        tokenizer: PreTrainedTokenizerFast,
        args: TrainerArguments,
        label_list: List[str],
        cls_token_at_end: Optional[bool] = False,
):
    """
    `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}
    logger.info(f"label_map: {label_map}")
    logger.info(f"id_to_label: {id_to_label}")
    logger.info(f"examples[0]: {examples[0]}")

    features = []
    for example in examples:
        # tokens = tokenizer.tokenize(example.text)
        # inputs = tokenizer._encode_plus(
        #     tokens,
        #     max_length=args.model.max_seq_length,
        #     truncation_strategy=TruncationStrategy.LONGEST_FIRST,
        #     padding_strategy=PaddingStrategy.MAX_LENGTH,
        # )

        example: NERExampleForKLUE = example
        logger.info(f"example.origin: {example.origin}")
        # inputs: BatchEncoding = tokenizer.__call__(example.origin,
        #                                               return_offsets_mapping=True,
        #                                               return_length=True, verbose=False,
        #                                               truncation=TruncationStrategy.DO_NOT_TRUNCATE,
        #                                               padding=PaddingStrategy.DO_NOT_PAD)
        # logger.info(f"batch_seq(1): {inputs}")
        inputs: BatchEncoding = tokenizer.__call__(example.origin,
                                                   return_offsets_mapping=True,
                                                   return_length=True, verbose=True,
                                                   max_length=args.model.max_seq_length,
                                                   truncation=TruncationStrategy.LONGEST_FIRST,
                                                   padding=PaddingStrategy.MAX_LENGTH)
        inputs2: List[str] = tokenizer.tokenize(example.origin, return_offsets_mapping=True, return_length=True, verbose=True)
        inputs3: BatchEncoding = tokenizer.encode_plus(example.origin, return_offsets_mapping=True, return_length=True, verbose=True)

        character_labels = {i: y for i, (_, y) in enumerate(example.character_list)}
        print(f"tokens: {inputs.tokens()}")
        print(f"character_labels: {character_labels}")

        for i in list(range(inputs['length'][0]))[:15]:
            span = inputs.token_to_chars(i)
            if span:
                sstr = example.origin[span.start:span.end]
                lable = _decide_span_label(character_labels[j] for j in range(span.start, span.end))
                print(i, sstr, span, lable)
        exit(1)

        logger.info(f"inputs(1): {type(inputs)} {inputs}")
        logger.info(f"inputs(2): {type(inputs2)} {inputs2}")
        logger.info(f"inputs(3): {type(inputs3)} {inputs3}")
        # inputs.char_to_token()
        print(len(inputs['input_ids']), inputs['input_ids'])
        print(len(inputs['token_type_ids']), inputs['token_type_ids'])
        print(len(inputs['attention_mask']), inputs['attention_mask'])
        print(len(inputs['offset_mapping']), inputs['offset_mapping'])

        print(inputs['length'])
        print([example.origin[a:b] for (a, b) in inputs['offset_mapping']])

        """
        target_sentence = "―<효진:PER> 역의 <김환희:PER>(<14:NOH>)가 특히 인상적이었다."
        tokens = ["―", "효", "##진", "역", "##의", "김", "##환", "##희",
                  "(", "14", ")", "가", "특히", "인상", "##적이", "##었다", "."]
        label_sequence = ['O', 'B-PER', 'I-PER', 'O', 'O', 'B-PER', 'I-PER', 'I-PER', 'O',
                          'B-NOH', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        """

        exit(1)
        label_ids = _process_target_sentence(
            tokens=tokens,
            origin_sentence=example.text,
            target_sentence=example.label,
            max_length=args.model.max_seq_length,
            label_map=label_map,
            tokenizer=tokenizer,
            cls_token_at_end=cls_token_at_end,
        )
        features.append(NERFeatures(**inputs, label_ids=label_ids))

    for i, example in enumerate(examples[:3]):
        logger.info("*** Example ***")
        logger.info("sentence: %s" % (example.text))
        logger.info("target: %s" % (example.label))
        logger.info("tokens: %s" % (" ".join(tokenizer.convert_ids_to_tokens(features[i].input_ids))))
        logger.info("label: %s" % (" ".join([id_to_label[label_id] for label_id in features[i].label_ids])))
        logger.info("features: %s" % features[i])

    return features


class NERDataset(Dataset):
    def __init__(
            self,
            split: str,
            args: TrainerArguments | TesterArguments,
            tokenizer: PreTrainedTokenizerFast,
            corpus: NERCorpus,
    ):
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
                logger.info("Saving features into cached file, it could take a lot of time...")
                torch.save(self.features, cache_data_path)
                logger.info(f"Saving features into cached file at {cache_data_path} [took {time.time() - start:.3f} s]")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.corpus.get_labels()


def parse_tagged(origin: str, tagged: str, debug: bool = False) -> Optional[NERExampleForKLUE]:
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
            parsed: Optional[NERExampleForKLUE] = parse_tagged(origin, tagged, debug=debug)
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
            parsed: Optional[NERExampleForKLUE] = parse_tagged(origin, tagged, debug=debug)
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
