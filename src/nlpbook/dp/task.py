import logging

import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from nlpbook.arguments import TesterArguments, TrainerArguments
from nlpbook.dp import DPDataset
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class DPTask(LightningModule):
    def __init__(self, model: PreTrainedModel,
                 args: TesterArguments | TrainerArguments,
                 trainer: pl.Trainer,
                 val_dataset: DPDataset,
                 total_steps: int,
                 ):
        super().__init__()
        self.model: PreTrainedModel = model
