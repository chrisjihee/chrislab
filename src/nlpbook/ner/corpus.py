import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from chrisbase.io import make_parent_dir
from nlpbook.arguments import TrainerArguments, TesterArguments
from transformers import BertTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

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

    def get_examples(self, data_path):
        logger.info(f"loading data from {data_path}...")
        examples = []
        for line in open(data_path, "r", encoding="utf-8").readlines():
            text, label = line.split("\u241E")
            examples.append(NERExample(text=text, label=label))
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


def _convert_examples_to_ner_features(
        examples: List[NERExample],
        tokenizer: BertTokenizer,
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

    features = []
    for example in examples:
        tokens = tokenizer.tokenize(example.text)
        inputs = tokenizer._encode_plus(
            tokens,
            max_length=args.model.max_seq_length,
            truncation_strategy=TruncationStrategy.LONGEST_FIRST,
            padding_strategy=PaddingStrategy.MAX_LENGTH,
        )
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
            tokenizer: BertTokenizer,
            corpus: NERCorpus,
            convert_examples_to_features_fn=_convert_examples_to_ner_features,
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
                self.features = convert_examples_to_features_fn(examples, tokenizer, args, label_list=self.corpus.get_labels())
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
