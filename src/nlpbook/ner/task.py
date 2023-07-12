import logging
from typing import Tuple, List, Dict

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from transformers import PreTrainedModel, CharSpan
from transformers.modeling_outputs import TokenClassifierOutput

from chrisbase.io import hr
from nlpbook.arguments import TesterArguments, TrainerArguments
from nlpbook.metrics import LabelMetricTool, accuracy, NER_Char_MacroF1, NER_Entity_MacroF1
from nlpbook.ner import NERDataset, NEREncodedExample

logger = logging.getLogger(__name__)


class NERTask(LightningModule):
    def __init__(self,
                 args: TesterArguments | TrainerArguments,
                 model: PreTrainedModel,
                 trainer: Trainer,
                 epoch_steps: int,
                 valid_dataset: NERDataset):
        super().__init__()
        self.args: TesterArguments | TrainerArguments = args
        self.model: PreTrainedModel = model
        self.trainer: Trainer = trainer
        self.epoch_steps: int = epoch_steps

        # for validation
        self.valid_dataset: NERDataset = valid_dataset
        self._labels: List[str] = self.valid_dataset.get_labels()
        self._label_to_id: Dict[str, int] = {label: i for i, label in enumerate(self._labels)}
        self._id_to_label: Dict[int, str] = {i: label for i, label in enumerate(self._labels)}
        self.metric_tools: Dict[str, LabelMetricTool] = {
            "F1c": NER_Char_MacroF1,
            "F1e": NER_Entity_MacroF1,
        }

        # initalize result
        self._valid_preds: List[int] = []
        self._valid_labels: List[int] = []
        self._valid_losses: List[torch.Tensor] = []
        self._train_losses: List[torch.Tensor] = []
        self._valid_accuracies: List[torch.Tensor] = []
        self._train_accuracies: List[torch.Tensor] = []

    def _global_step(self) -> float:
        return self.trainer.lightning_module.global_step * 1.0

    def _global_epoch(self) -> float:
        return self._global_step() / self.epoch_steps

    def _learning_rate(self) -> float:
        return self.trainer.optimizers[0].param_groups[0]["lr"]

    def _train_loss(self) -> torch.Tensor:
        return torch.tensor(self._train_losses).mean()

    def _valid_loss(self) -> torch.Tensor:
        return torch.tensor(self._valid_losses).mean()

    def _train_accuracy(self) -> torch.Tensor:
        return torch.tensor(self._train_accuracies).mean()

    def _valid_accuracy(self) -> torch.Tensor:
        return torch.tensor(self._valid_accuracies).mean()

    def _valid_metric(self, metric_tool: LabelMetricTool) -> torch.Tensor | float:
        metric_tool.reset()
        metric_tool.update(self._valid_preds, self._valid_labels, self._labels)
        return metric_tool.compute()

    def _log_value(self, name: str, value: torch.Tensor | float):
        self.log(name, value, batch_size=self.args.hardware.batch_size, sync_dist=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        batch.pop("example_ids")
        outputs: TokenClassifierOutput = self.model(**batch)
        labels: torch.Tensor = batch["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds, labels, ignore_index=0)
        return {
            "loss": outputs.loss,
            "acc": acc,
        }

    @staticmethod
    def label_to_char_labels(label, num_char):
        for i in range(num_char):
            if i > 0 and ("-" in label):
                yield "I-" + label.split("-", maxsplit=1)[-1]
            else:
                yield label

    def label_to_id(self, x):
        return self._label_to_id[x]

    def id_to_label(self, x):
        return self._id_to_label[x]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor | List[int]]:
        if self.args.env.debugging:
            logger.debug('')
            logger.debug(f"[validation_step] batch_idx: {batch_idx}, global_step: {self.trainer.lightning_module.global_step}")
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    logger.debug(f"  - batch[{key:14s}]     = {batch[key].shape} | {batch[key].tolist()}")
                else:
                    logger.debug(f"  - batch[{key:14s}]     = ({len(batch[key])}) | {batch[key]}")
        example_ids: List[int] = batch.pop("example_ids").tolist()
        outputs: TokenClassifierOutput = self.model(**batch)
        labels: torch.Tensor = batch["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds=preds, labels=labels, ignore_index=0)

        dict_of_token_pred_ids: Dict[int, List[int]] = {}
        dict_of_char_label_ids: Dict[int, List[int]] = {}
        dict_of_char_pred_ids: Dict[int, List[int]] = {}
        for token_pred_ids, example_id in zip(preds.tolist(), example_ids):
            token_pred_tags: List[str] = [self.id_to_label(x) for x in token_pred_ids]
            encoded_example: NEREncodedExample = self.valid_dataset[example_id]
            offset_to_label: Dict[int, str] = encoded_example.raw.get_offset_label_dict()
            all_char_pair_tags: List[Tuple[str | None, str | None]] = [(None, None)] * len(encoded_example.raw.character_list)
            for token_id in range(self.args.model.seq_len):
                token_span: CharSpan = encoded_example.encoded.token_to_chars(token_id)
                if token_span:
                    char_pred_tags = NERTask.label_to_char_labels(token_pred_tags[token_id], token_span.end - token_span.start)
                    for offset, char_pred_tag in zip(range(token_span.start, token_span.end), char_pred_tags):
                        all_char_pair_tags[offset] = (offset_to_label[offset], char_pred_tag)
            valid_char_pair_tags = [(a, b) for a, b in all_char_pair_tags if a and b]
            valid_char_label_ids = [self.label_to_id(a) for a, b in valid_char_pair_tags]
            valid_char_pred_ids = [self.label_to_id(b) for a, b in valid_char_pair_tags]
            dict_of_token_pred_ids[example_id] = token_pred_ids
            dict_of_char_label_ids[example_id] = valid_char_label_ids
            dict_of_char_pred_ids[example_id] = valid_char_pred_ids

        if self.args.env.debugging:
            logger.debug(hr())
        list_of_char_pred_ids: List[int] = []
        list_of_char_label_ids: List[int] = []
        for encoded_example in [self.valid_dataset[i] for i in example_ids]:
            char_label_ids = dict_of_char_label_ids[encoded_example.idx]
            char_pred_ids = dict_of_char_pred_ids[encoded_example.idx]
            assert len(char_pred_ids) == len(char_label_ids)
            list_of_char_pred_ids.extend(char_pred_ids)
            list_of_char_label_ids.extend(char_label_ids)
            if self.args.env.debugging:
                token_pred_ids = dict_of_token_pred_ids[encoded_example.idx]
                logger.debug(f"  - encoded_example.idx                = {encoded_example.idx}")
                logger.debug(f"  - encoded_example.raw.entity_list    = ({len(encoded_example.raw.entity_list)}) {encoded_example.raw.entity_list}")
                logger.debug(f"  - encoded_example.raw.origin         = ({len(encoded_example.raw.origin)}) {encoded_example.raw.origin}")
                logger.debug(f"  - encoded_example.raw.character_list = ({len(encoded_example.raw.character_list)}) {' | '.join(f'{x}/{y}' for x, y in encoded_example.raw.character_list)}")
                logger.debug(f"  - encoded_example.encoded.tokens()   = ({len(encoded_example.encoded.tokens())}) {' '.join(encoded_example.encoded.tokens())}")

                def id_label(x):
                    return f"{self.id_to_label(x):5s}"

                logger.debug(f"  - encoded_example.label_ids          = ({len(encoded_example.label_ids)}) {' '.join(map(str, map(id_label, encoded_example.label_ids)))}")
                logger.debug(f"  - encoded_example.token_pred_ids     = ({len(token_pred_ids)}) {' '.join(map(str, map(id_label, token_pred_ids)))}")
                logger.debug(f"  - encoded_example.char_label_ids     = ({len(char_label_ids)}) {' '.join(map(str, map(id_label, char_label_ids)))}")
                logger.debug(f"  - encoded_example.char_pred_ids      = ({len(char_pred_ids)}) {' '.join(map(str, map(id_label, char_pred_ids)))}")
                logger.debug(hr('-'))
        assert len(list_of_char_pred_ids) == len(list_of_char_label_ids)

        if self.args.env.debugging:
            def id_str(x):
                return f"{x:02d}"

            logger.debug(f"  - list_of_char_label_ids = ({len(list_of_char_label_ids)}) {' '.join(map(str, map(id_str, list_of_char_label_ids)))}")
            logger.debug(f"  - list_of_char_pred_ids  = ({len(list_of_char_pred_ids)}) {' '.join(map(str, map(id_str, list_of_char_pred_ids)))}")
        return {
            "loss": outputs.loss,
            "acc": acc,
            "preds": list_of_char_pred_ids,
            "labels": list_of_char_label_ids
        }

    def on_train_epoch_start(self) -> None:
        self._train_losses.clear()
        self._train_accuracies.clear()

    def on_validation_epoch_start(self) -> None:
        self._valid_preds.clear()
        self._valid_labels.clear()
        self._valid_losses.clear()
        self._valid_accuracies.clear()

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._train_losses.append(outputs["loss"])
        self._train_accuracies.append(outputs["acc"])
        self._log_value("g_step", self._global_step())
        self._log_value("g_epoch", self._global_epoch())
        self._log_value("lr", self._learning_rate())
        self._log_value("loss", outputs["loss"])
        self._log_value("acc", outputs["acc"])
        self._log_value("avg_loss", self._train_loss())
        self._log_value("avg_acc", self._train_accuracy())

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor | List[int]], batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._valid_preds.extend(outputs["preds"])
        self._valid_labels.extend(outputs["labels"])
        self._valid_losses.append(outputs["loss"])
        self._valid_accuracies.append(outputs["acc"])

    def on_validation_epoch_end(self) -> None:
        assert self._valid_preds
        assert self._valid_labels
        assert len(self._valid_preds) == len(self._valid_labels)
        self._log_value("g_step", self._global_step())
        self._log_value("g_epoch", self._global_epoch())
        self._log_value("lr", self._learning_rate())
        self._log_value("avg_loss", self._train_loss())
        self._log_value("avg_acc", self._train_accuracy())
        self._log_value("val_loss", self._valid_loss())
        self._log_value("val_acc", self._valid_accuracy())
        for name, tool in self.metric_tools.items():
            self._log_value(f"val_{name}", self._valid_metric(tool))
        self.on_validation_epoch_start()
        self.on_train_epoch_start()  # reset accumulated train losses after validation
