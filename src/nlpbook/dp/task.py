import logging
from typing import Dict, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from nlpbook.arguments import TesterArguments, TrainerArguments
from nlpbook.dp import ModelForDependencyParsing
from nlpbook.metrics import DPResult

logger = logging.getLogger(__name__)


class DPTask(LightningModule):
    def __init__(self,
                 args: TesterArguments | TrainerArguments,
                 model: ModelForDependencyParsing,
                 trainer: Trainer):
        super().__init__()
        self.args: TesterArguments | TrainerArguments = args
        self.model: ModelForDependencyParsing = model
        self.trainer: Trainer = trainer

        # initialize setting
        self._log_kwargs = {
            "batch_size": self.args.hardware.batch_size,
            "sync_dist": True,
            "prog_bar": True,
            "logger": True,
        }

        # initalize result
        self._valid_preds: List[DPResult] = []
        self._valid_labels: List[DPResult] = []
        self._valid_losses: List[torch.Tensor] = []
        self._train_losses: List[torch.Tensor] = []

    def _learning_rate(self):
        return self.trainer.optimizers[0].param_groups[0]["lr"]

    def _train_loss(self):
        return torch.tensor(self._train_losses).mean()

    def _valid_loss(self):
        return torch.tensor(self._valid_losses).mean()

    def _valid_metric(self, tool):
        tool.reset()
        tool.update(self._valid_preds, self._valid_labels)
        return tool.compute()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch.pop("example_ids")
        inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

        batch_size = batch["head_ids"].size()[0]
        batch_index = torch.arange(0, int(batch_size)).long()
        max_word_length = batch["max_word_length"].item()
        head_index = (
            torch.arange(0, max_word_length).view(max_word_length, 1).expand(max_word_length, batch_size).long()
        )

        # forward
        out_arc, out_type = self.model.forward(
            batch["bpe_head_mask"],
            batch["bpe_tail_mask"],
            batch["pos_ids"],
            batch["head_ids"],
            max_word_length,
            batch["mask_e"],
            batch["mask_d"],
            batch_index,
            **inputs,
        )

        # compute loss
        minus_inf = -1e8
        minus_mask_d = (1 - batch["mask_d"]) * minus_inf
        minus_mask_e = (1 - batch["mask_e"]) * minus_inf
        out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

        loss_arc = F.log_softmax(out_arc, dim=2)
        loss_type = F.log_softmax(out_type, dim=2)

        loss_arc = loss_arc * batch["mask_d"].unsqueeze(2) * batch["mask_e"].unsqueeze(1)
        loss_type = loss_type * batch["mask_d"].unsqueeze(2)
        num = batch["mask_d"].sum()

        loss_arc = loss_arc[batch_index, head_index, batch["head_ids"].data.t()].transpose(0, 1)
        loss_type = loss_type[batch_index, head_index, batch["type_ids"].data.t()].transpose(0, 1)
        loss_arc = -loss_arc.sum() / num
        loss_type = -loss_type.sum() / num
        loss = loss_arc + loss_type

        # accumulate result
        self._train_losses.append(loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch.pop("example_ids")
        inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

        batch_size = batch["head_ids"].size()[0]
        batch_index = torch.arange(0, int(batch_size)).long()
        max_word_length = batch["max_word_length"].item()
        head_index = (
            torch.arange(0, max_word_length).view(max_word_length, 1).expand(max_word_length, batch_size).long()
        )

        # forward
        out_arc, out_type = self.model.forward(
            batch["bpe_head_mask"],
            batch["bpe_tail_mask"],
            batch["pos_ids"],
            batch["head_ids"],
            max_word_length,
            batch["mask_e"],
            batch["mask_d"],
            batch_index,
            is_training=False,
            **inputs,
        )

        # compute loss
        minus_inf = -1e8
        minus_mask_d = (1 - batch["mask_d"]) * minus_inf
        minus_mask_e = (1 - batch["mask_e"]) * minus_inf
        out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

        loss_arc = F.log_softmax(out_arc, dim=2)
        loss_type = F.log_softmax(out_type, dim=2)

        loss_arc = loss_arc * batch["mask_d"].unsqueeze(2) * batch["mask_e"].unsqueeze(1)
        loss_type = loss_type * batch["mask_d"].unsqueeze(2)
        num = batch["mask_d"].sum()

        loss_arc = loss_arc[batch_index, head_index, batch["head_ids"].data.t()].transpose(0, 1)
        loss_type = loss_type[batch_index, head_index, batch["type_ids"].data.t()].transpose(0, 1)
        loss_arc = -loss_arc.sum() / num
        loss_type = -loss_type.sum() / num
        loss = loss_arc + loss_type

        # predict arc and its type
        pred_heads: torch.Tensor = torch.argmax(out_arc, dim=2)
        pred_types: torch.Tensor = torch.argmax(out_type, dim=2)
        preds = DPResult(pred_heads, pred_types)
        labels = DPResult(batch["head_ids"], batch["type_ids"])

        # accumulate result
        self._valid_preds.append(preds)
        self._valid_labels.append(labels)
        self._valid_losses.append(loss)
        return loss

    def on_train_epoch_start(self) -> None:
        self._train_losses.clear()

    def on_validation_epoch_start(self) -> None:
        self._valid_preds.clear()
        self._valid_labels.clear()
        self._valid_losses.clear()

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self.log("loss", outputs["loss"], **self._log_kwargs)

    def on_validation_epoch_end(self) -> None:
        assert len(self._valid_preds) == len(self._valid_labels)
        self.log("lr", self._learning_rate(), **self._log_kwargs)
        self.log("avg_loss", self._train_loss(), **self._log_kwargs)
        self.log("val_loss", self._valid_loss(), **self._log_kwargs)
        for name, tool in self.model.metric_tools.items():
            self.log(f"val_{name}", self._valid_metric(tool), **self._log_kwargs)
        self._valid_preds.clear()
        self._valid_labels.clear()
        self._valid_losses.clear()
        self._train_losses.clear()
