from typing import List, Dict

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from nlpbook.arguments import TrainerArguments, TesterArguments
from nlpbook.metrics import accuracy


class ClassificationTask(LightningModule):
    def __init__(self,
                 args: TrainerArguments | TesterArguments,
                 model: PreTrainedModel,
                 trainer: Trainer,
                 epoch_steps: int):
        super().__init__()
        self.args: TesterArguments | TrainerArguments = args
        self.model: PreTrainedModel = model
        self.trainer: Trainer = trainer
        self.epoch_steps: int = epoch_steps

        # initalize result
        self._valid_preds: List[int] = []
        self._valid_labels: List[int] = []
        self._valid_losses: List[torch.Tensor] = []
        self._valid_accuracies: List[torch.Tensor] = []

        # initalize result
        self._test_preds: List[int] = []
        self._test_labels: List[int] = []
        self._test_losses: List[torch.Tensor] = []
        self._test_accuracies: List[torch.Tensor] = []

    def _global_step(self) -> float:
        return self.trainer.lightning_module.global_step * 1.0

    def _global_epoch(self) -> float:
        return self._global_step() / self.epoch_steps

    def _learning_rate(self) -> float:
        return self.trainer.optimizers[0].param_groups[0]["lr"]

    def _valid_loss(self) -> torch.Tensor:
        return torch.tensor(self._valid_losses).mean()

    def _valid_accuracy(self) -> torch.Tensor:
        return torch.tensor(self._valid_accuracies).mean()

    def _test_loss(self) -> torch.Tensor:
        return torch.tensor(self._test_losses).mean()

    def _test_accuracy(self) -> torch.Tensor:
        return torch.tensor(self._test_accuracies).mean()

    def _log_value(self, name: str, value: torch.Tensor | float):
        self.log(name, value, batch_size=self.args.hardware.batch_size, sync_dist=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def training_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        labels: torch.Tensor = inputs["labels"]
        acc: torch.Tensor = accuracy(preds, labels)
        return {
            "loss": outputs.loss,
            "acc": acc,
        }

    def validation_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        labels: torch.Tensor = inputs["labels"]
        acc: torch.Tensor = accuracy(preds, labels)
        return {
            "loss": outputs.loss,
            "acc": acc,
            "preds": preds,
            "labels": labels
        }

    def test_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels)
        return {
            "loss": outputs.loss,
            "acc": acc,
            "preds": preds,
            "labels": labels
        }

    def on_validation_epoch_start(self) -> None:
        self._valid_preds.clear()
        self._valid_labels.clear()
        self._valid_losses.clear()
        self._valid_accuracies.clear()

    def on_test_epoch_start(self) -> None:
        self._test_preds.clear()
        self._test_labels.clear()
        self._test_losses.clear()
        self._test_accuracies.clear()

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._log_value("g_step", self._global_step())
        self._log_value("g_epoch", self._global_epoch())
        self._log_value("lr", self._learning_rate())
        self._log_value("loss", outputs["loss"])
        self._log_value("acc", outputs["acc"])

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor | List[int]], batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._valid_preds.extend(outputs["preds"])
        self._valid_labels.extend(outputs["labels"])
        self._valid_losses.append(outputs["loss"])
        self._valid_accuracies.append(outputs["acc"])

    def on_test_batch_end(self, outputs: Dict[str, torch.Tensor | List[int]], batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._test_preds.extend(outputs["preds"])
        self._test_labels.extend(outputs["labels"])
        self._test_losses.append(outputs["loss"])
        self._test_accuracies.append(outputs["acc"])

    def on_validation_epoch_end(self) -> None:
        assert self._valid_preds
        assert self._valid_labels
        assert len(self._valid_preds) == len(self._valid_labels)
        self._log_value("g_step", self._global_step())
        self._log_value("g_epoch", self._global_epoch())
        self._log_value("lr", self._learning_rate())
        self._log_value("val_loss", self._valid_loss())
        self._log_value("val_acc", self._valid_accuracy())
        self.on_validation_epoch_start()  # reset accumulated values

    def on_test_epoch_end(self) -> None:
        assert self._test_preds
        assert self._test_labels
        assert len(self._test_preds) == len(self._test_labels)
        self._log_value("g_step", self._global_step())
        self._log_value("g_epoch", self._global_epoch())
        self._log_value("lr", self._learning_rate())
        self._log_value("test_loss", self._test_loss())
        self._log_value("test_acc", self._test_accuracy())
        self.on_test_epoch_start()  # reset accumulated values
