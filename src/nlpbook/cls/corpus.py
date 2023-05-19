import csv
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from nlpbook.arguments import TrainerArguments, TesterArguments
from transformers import PreTrainedTokenizer

logger = logging.getLogger("nlpbook")


@dataclass
class ClassificationExample:
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None


@dataclass
class ClassificationFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[int] = None


class NsmcCorpus:

    def __init__(self):
        pass

    def get_examples(self, data_path):
        logger.info(f"loading data from {data_path}...")
        lines = list(csv.reader(open(data_path, "r", encoding="utf-8"), delimiter="\t", quotechar='"'))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            _, text_a, label = line
            examples.append(ClassificationExample(text_a=text_a, text_b=None, label=label))
        return examples

    def get_labels(self):
        return ["0", "1"]

    @property
    def num_labels(self):
        return len(self.get_labels())


def _convert_examples_to_cls_features(
        examples: List[ClassificationExample],
        tokenizer: PreTrainedTokenizer,
        args: TrainerArguments,
        label_list: List[str],
):
    label_map = {label: i for i, label in enumerate(label_list)}
    labels = [label_map[example.label] for example in examples]

    logger.info("tokenize sentences, it could take a lot of time...")
    start = time.time()
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=args.model.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    logger.info("tokenize sentences [took %.3f s]", time.time() - start)

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = ClassificationFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:3]):
        logger.info("*** Example ***")
        if example.text_b is None:
            logger.info("sentence: %s" % (example.text_a))
        else:
            sentence = example.text_a + " + " + example.text_b
            logger.info("sentence A, B: %s" % (sentence))
        logger.info("tokens: %s" % (" ".join(tokenizer.convert_ids_to_tokens(features[i].input_ids))))
        logger.info("label: %s" % (example.label))
        logger.info("features: %s" % features[i])

    return features


class ClassificationDataset(Dataset):

    def __init__(
            self,
            split: str,
            args: TrainerArguments | TesterArguments,
            tokenizer: PreTrainedTokenizer,
            corpus: NsmcCorpus,
            convert_examples_to_features_fn=_convert_examples_to_cls_features,
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
            .with_stem(text_data_path.stem + f"-by-{tokenizer.__class__.__name__}-as-{args.model.max_seq_length}seq") \
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
