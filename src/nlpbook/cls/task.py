import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from nlpbook.arguments import TrainerArguments, TesterArguments
from nlpbook.metrics import accuracy
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ClassificationTask(LightningModule):
    def __init__(self, model: PreTrainedModel,
                 args: TrainerArguments | TesterArguments,
                 trainer: pl.Trainer):
        super().__init__()
        self.model = model
        self.args = args
        self.trainer = trainer
        self.train_acc = -1.0
        self.train_loss = -1.0

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning.speed)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def training_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels)
        self.train_acc = acc
        self.train_loss = outputs.loss
        return {"loss": outputs.loss, "acc": acc}

    def validation_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels)
        global_step = self.trainer.lightning_module.global_step * 1.0
        self.log(prog_bar=True, logger=False, on_epoch=True, name="global_step", value=global_step)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="train_loss", value=self.train_loss)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="train_acc", value=self.train_acc)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="val_loss", value=outputs.loss)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="val_acc", value=acc)
        return {"val_loss": outputs.loss, "val_acc": acc}

    def test_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels)
        self.log(prog_bar=True, logger=True, on_epoch=True, name="test_loss", value=outputs.loss)
        self.log(prog_bar=True, logger=True, on_epoch=True, name="test_acc", value=acc)
        return {"test_loss": outputs.loss, "test_acc": acc}
