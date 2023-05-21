from typing import Any, List

import numpy as np
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.scheme import IOB2
from sklearn import metrics


def klue_ner_entity_macro_f1(preds: List[int], labels: List[int], label_list: List[str]) -> Any:
    """KLUE-NER entity-level macro F1 (except O tag)"""
    preds = np.array(preds).flatten().tolist()
    labels = np.array(labels).flatten().tolist()
    preds_label = []
    labels_label = []

    for pred in preds:
        preds_label.append(label_list[pred])
    for label in labels:
        labels_label.append(label_list[label])

    entity_macro_f1 = ner_f1_score([labels_label], [preds_label], average="macro", mode="strict", scheme=IOB2)
    return entity_macro_f1 * 100.0


def klue_ner_char_macro_f1(preds: List[int], labels: List[int], label_list: List[str]) -> Any:
    """KLUE-NER character level macro f1 (except O tag)"""
    label_indices = list(range(len(label_list)))
    preds = np.array(preds).flatten().tolist()
    trues = np.array(labels).flatten().tolist()
    return metrics.f1_score(trues, preds, labels=label_indices, average="macro", zero_division=True) * 100.0
