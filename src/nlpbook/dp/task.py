import logging
from typing import List, Dict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from nlpbook.arguments import TesterArguments, TrainerArguments
from nlpbook.dp import ModelForDependencyParsing
from nlpbook.metrics import DPResult, BasicMetricTool, DP_UAS_MacroF1, DP_UAS_MicroF1, DP_LAS_MacroF1, DP_LAS_MicroF1

logger = logging.getLogger(__name__)


class DPTask(LightningModule):
    def __init__(self,
                 args: TesterArguments | TrainerArguments,
                 model: ModelForDependencyParsing,
                 trainer: Trainer,
                 epoch_steps: int):
        super().__init__()
        self.args: TesterArguments | TrainerArguments = args
        self.model: ModelForDependencyParsing = model
        self.trainer: Trainer = trainer
        self.epoch_steps: int = epoch_steps

        # for validation
        self.metric_tools: Dict[str, BasicMetricTool] = {
            "UASa": DP_UAS_MacroF1,
            "UASi": DP_UAS_MicroF1,
            "LASa": DP_LAS_MacroF1,
            "LASi": DP_LAS_MicroF1,
        }

        # initalize result
        self._valid_preds: List[DPResult] = []
        self._valid_labels: List[DPResult] = []
        self._valid_losses: List[torch.Tensor] = []
        self._train_losses: List[torch.Tensor] = []

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

    def _valid_metric(self, metric_tool: BasicMetricTool) -> torch.Tensor | float:
        metric_tool.reset()
        metric_tool.update(self._valid_preds, self._valid_labels)
        # logger.info("")
        # logger.info("")
        # logger.info("")
        # logger.info(f"self._valid_preds: {self._valid_preds}")
        # logger.info(f"self._valid_labels: {self._valid_labels}")
        # logger.info(f"metric_tool.compute()={metric_tool.compute()}")
        # DP_UAS_MacroF1.reset()
        # DP_UAS_MacroF1.update(self._valid_preds, self._valid_labels)
        # logger.info(f'DP_UAS_MacroF1.compute()={DP_UAS_MacroF1.compute()}')
        # exit(1)
        return metric_tool.compute()

    def _log_value(self, name: str, value: torch.Tensor | float):
        self.log(name, value, batch_size=self.args.hardware.train_batch, sync_dist=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
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
        return {
            "loss": loss,
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor | DPResult]:
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
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
        }

    def on_train_epoch_start(self) -> None:
        self._train_losses.clear()

    def on_validation_epoch_start(self) -> None:
        self._valid_preds.clear()
        self._valid_labels.clear()
        self._valid_losses.clear()

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._train_losses.append(outputs["loss"])
        self._log_value("g_step", self._global_step())
        self._log_value("g_epoch", self._global_epoch())
        self._log_value("lr", self._learning_rate())
        self._log_value("loss", outputs["loss"])
        self._log_value("avg_loss", self._train_loss())

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor | DPResult], batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._valid_preds.append(outputs["preds"])
        self._valid_labels.append(outputs["labels"])
        self._valid_losses.append(outputs["loss"])

    def on_validation_epoch_end(self) -> None:
        assert self._valid_preds
        assert self._valid_labels
        assert len(self._valid_preds) == len(self._valid_labels)
        self._log_value("g_step", self._global_step())
        self._log_value("g_epoch", self._global_epoch())
        self._log_value("lr", self._learning_rate())
        self._log_value("avg_loss", self._train_loss())
        self._log_value("val_loss", self._valid_loss())
        for name, tool in self.metric_tools.items():
            self._log_value(f"val_{name}", self._valid_metric(tool))
        self.on_validation_epoch_start()
        self.on_train_epoch_start()  # reset accumulated train losses after validation
