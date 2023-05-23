from typing import List, Dict, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from chrisbase.io import out_hr
from klue_baseline.metrics.functional import klue_ner_entity_macro_f1, klue_ner_char_macro_f1
from nlpbook.arguments import TesterArguments, TrainerArguments
from nlpbook.metrics import accuracy
from nlpbook.ner import NERDataset, NEREncodedExample
from transformers import PreTrainedModel, CharSpan
from transformers.modeling_outputs import TokenClassifierOutput


def label_to_char_labels(label, num_char):
    for i in range(num_char):
        if i > 0 and ("-" in label):
            yield "I-" + label.split("-", maxsplit=1)[-1]
        else:
            yield label


class NERTask(LightningModule):
    def __init__(self, model: PreTrainedModel,
                 args: TesterArguments | TrainerArguments,
                 trainer: pl.Trainer,
                 val_dataset: NERDataset,
                 total_steps: int,
                 ):
        super().__init__()
        self.model: PreTrainedModel = model
        self.args: TesterArguments | TrainerArguments = args
        self.trainer: pl.Trainer = trainer

        self.val_dataset: NERDataset = val_dataset
        self._validation_char_pred_ids = None
        self._validation_char_label_ids = None

        self._labels: List[str] = self.val_dataset.get_labels()
        self._label_to_id: Dict[str, int] = {label: i for i, label in enumerate(self._labels)}
        self._id_to_label: Dict[int, str] = {i: label for i, label in enumerate(self._labels)}

        self.total_steps: int = total_steps
        self.train_loss: float = -1.0
        self.train_acc: float = -1.0

    def get_labels(self):
        return self._labels

    def label_to_id(self, x):
        return self._label_to_id[x]

    def id_to_label(self, x):
        return self._id_to_label[x]

    def _global_step(self):
        return self.trainer.lightning_module.global_step

    def _trained_rate(self):
        return self.trainer.lightning_module.global_step / self.total_steps

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning.speed)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def training_step(self, batch: Dict[str, torch.Tensor | List[int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        _: List[int] = batch.pop("example_ids")
        outputs: TokenClassifierOutput = self.model(**batch)
        labels: torch.Tensor = batch["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds, labels, ignore_index=0)
        self.train_loss = outputs.loss
        self.train_acc = acc
        return outputs.loss

    def validation_step(self, batch: Dict[str, torch.Tensor | List[int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.args.env.on_debugging:
            print()
            print(f"[validation_step] batch_idx: {batch_idx}, global_step: {self._global_step()}")
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    print(f"  - batch[{key:14s}]     = {batch[key].shape} | {batch[key].tolist()}")
                else:
                    print(f"  - batch[{key:14s}]     = ({len(batch[key])}) | {batch[key]}")
        example_ids: List[int] = batch.pop("example_ids")
        outputs: TokenClassifierOutput = self.model(**batch)
        labels: torch.Tensor = batch["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds, labels, ignore_index=0)

        self.log(prog_bar=True, logger=False, on_epoch=True, name="global_step", value=self._global_step() * 1.0)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="trained_rate", value=self._trained_rate())
        self.log(prog_bar=True, logger=False, on_epoch=True, name="train_loss", value=self.train_loss)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="train_acc", value=self.train_acc)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="val_loss", value=outputs.loss)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="val_acc", value=acc)

        dict_of_token_pred_ids: Dict[int, List[int]] = {}
        dict_of_char_label_ids: Dict[int, List[int]] = {}
        dict_of_char_pred_ids: Dict[int, List[int]] = {}
        for token_pred_ids, example_id in zip(preds.tolist(), example_ids):
            token_pred_tags: List[str] = [self.id_to_label(x) for x in token_pred_ids]
            encoded_example: NEREncodedExample = self.val_dataset[example_id]
            offset_to_label: Dict[int, str] = encoded_example.raw.get_offset_label_dict()
            all_char_pair_tags: List[Tuple[str | None, str | None]] = [(None, None)] * len(encoded_example.raw.character_list)
            for token_id in range(self.args.model.max_seq_length):
                token_span: CharSpan = encoded_example.encoded.token_to_chars(token_id)
                if token_span:
                    char_pred_tags = label_to_char_labels(token_pred_tags[token_id], token_span.end - token_span.start)
                    for offset, char_pred_tag in zip(range(token_span.start, token_span.end), char_pred_tags):
                        all_char_pair_tags[offset] = (offset_to_label[offset], char_pred_tag)
            valid_char_pair_tags = [(a, b) for a, b in all_char_pair_tags if a and b]
            valid_char_label_ids = [self.label_to_id(a) for a, b in valid_char_pair_tags]
            valid_char_pred_ids = [self.label_to_id(b) for a, b in valid_char_pair_tags]
            dict_of_token_pred_ids[example_id] = token_pred_ids
            dict_of_char_label_ids[example_id] = valid_char_label_ids
            dict_of_char_pred_ids[example_id] = valid_char_pred_ids

        if self.args.env.on_tracing:
            out_hr()
        flatlist_of_char_pred_ids: List[int] = []
        flatlist_of_char_label_ids: List[int] = []
        for encoded_example in [self.val_dataset[i] for i in example_ids]:
            token_pred_ids = dict_of_token_pred_ids[encoded_example.idx]
            char_label_ids = dict_of_char_label_ids[encoded_example.idx]
            char_pred_ids = dict_of_char_pred_ids[encoded_example.idx]
            flatlist_of_char_pred_ids.extend(char_pred_ids)
            flatlist_of_char_label_ids.extend(char_label_ids)
            if self.args.env.on_tracing:
                print(f"  - encoded_example.idx                = {encoded_example.idx}")
                print(f"  - encoded_example.raw.entity_list    = ({len(encoded_example.raw.entity_list)}) {encoded_example.raw.entity_list}")
                print(f"  - encoded_example.raw.origin         = ({len(encoded_example.raw.origin)}) {encoded_example.raw.origin}")
                print(f"  - encoded_example.raw.character_list = ({len(encoded_example.raw.character_list)}) {' | '.join(f'{x}/{y}' for x, y in encoded_example.raw.character_list)}")
                print(f"  - encoded_example.encoded.tokens()   = ({len(encoded_example.encoded.tokens())}) {' '.join(encoded_example.encoded.tokens())}")
                current_repr = lambda x: f"{self.id_to_label(x):5s}"
                print(f"  - encoded_example.label_ids          = ({len(encoded_example.label_ids)}) {' '.join(map(str, map(current_repr, encoded_example.label_ids)))}")
                print(f"  - encoded_example.token_pred_ids     = ({len(token_pred_ids)}) {' '.join(map(str, map(current_repr, token_pred_ids)))}")
                print(f"  - encoded_example.char_label_ids     = ({len(char_label_ids)}) {' '.join(map(str, map(current_repr, char_label_ids)))}")
                print(f"  - encoded_example.char_pred_ids      = ({len(char_pred_ids)}) {' '.join(map(str, map(current_repr, char_pred_ids)))}")
                out_hr('-')

        if self.args.env.on_debugging:
            current_repr = lambda x: f"{x:02d}"
            print(f"  - flatlist_of_char_label_ids = ({len(flatlist_of_char_label_ids)}) {' '.join(map(str, map(current_repr, flatlist_of_char_label_ids)))}")
            print(f"  - flatlist_of_char_pred_ids  = ({len(flatlist_of_char_pred_ids)}) {' '.join(map(str, map(current_repr, flatlist_of_char_pred_ids)))}")
        assert len(flatlist_of_char_label_ids) == len(flatlist_of_char_pred_ids)
        self._validation_char_pred_ids.extend(flatlist_of_char_pred_ids)
        self._validation_char_label_ids.extend(flatlist_of_char_label_ids)
        return outputs.loss

    def on_validation_start(self):
        self._validation_char_pred_ids: List[int] = []
        self._validation_char_label_ids: List[int] = []

    def on_validation_epoch_end(self):
        assert self._validation_char_pred_ids and self._validation_char_label_ids
        assert len(self._validation_char_pred_ids) == len(self._validation_char_label_ids)
        chr_f1 = klue_ner_char_macro_f1(preds=self._validation_char_pred_ids, labels=self._validation_char_label_ids, label_list=self._labels)
        ent_f1 = klue_ner_entity_macro_f1(preds=self._validation_char_pred_ids, labels=self._validation_char_label_ids, label_list=self._labels)
        self.log(prog_bar=True, logger=True, on_epoch=True, name="chr_f1", value=chr_f1)
        self.log(prog_bar=True, logger=True, on_epoch=True, name="ent_f1", value=ent_f1)

    def test_step(self, batch, batch_idx):
        outputs: TokenClassifierOutput = self.model(**batch)
        labels: torch.Tensor = batch["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds, labels, ignore_index=0)
        self.log(prog_bar=False, logger=True, on_epoch=True, name="test_loss", value=outputs.loss)
        self.log(prog_bar=False, logger=True, on_epoch=True, name="test_acc", value=acc)
        return outputs.loss
