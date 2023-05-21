from typing import List, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from chrisbase.io import out_hr
from nlpbook.arguments import TrainerArguments, TesterArguments
from nlpbook.metrics import accuracy
from nlpbook.ner import NER_PAD_ID, NERDataset, NERRawExample, NEREncodedExample
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class NERTask(LightningModule):
    def __init__(self, model: PreTrainedModel,
                 args: TrainerArguments | TesterArguments,
                 trainer: pl.Trainer,
                 val_dataset: NERDataset,
                 total_steps: int,
                 ):
        super().__init__()
        self.model = model
        self.args = args
        self.trainer = trainer
        self.val_dataset = val_dataset
        self.total_steps = total_steps
        self.train_loss = -1.0
        self.train_acc = -1.0

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
        print()
        print(f"[training_step] batch_idx: {batch_idx}, global_step: {self._global_step()}")
        _: List[int] = batch.pop("example_ids")
        outputs: TokenClassifierOutput = self.model(**batch)
        labels: torch.Tensor = batch["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds, labels, ignore_index=NER_PAD_ID)
        self.train_loss = outputs.loss
        self.train_acc = acc
        return {"loss": outputs.loss}

    def validation_step(self, batch: Dict[str, torch.Tensor | List[int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        print()
        print(f"[validation_step] batch_idx: {batch_idx}, global_step: {self._global_step()}")
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                print(f"  - batch[{key:14s}]     = {batch[key].shape} | {batch[key].tolist()}")
            else:
                print(f"  - batch[{key:14s}]     = {batch[key]}")
        example_ids: List[int] = batch.pop("example_ids")
        outputs: TokenClassifierOutput = self.model(**batch)
        labels: torch.Tensor = batch["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds, labels, ignore_index=NER_PAD_ID)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="global_step", value=self._global_step() * 1.0)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="trained_rate", value=self._trained_rate())
        self.log(prog_bar=True, logger=False, on_epoch=True, name="train_loss", value=self.train_loss)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="train_acc", value=self.train_acc)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="val_loss", value=outputs.loss)
        self.log(prog_bar=True, logger=False, on_epoch=True, name="val_acc", value=acc)

        out_hr()
        print(f"  -                     preds = {preds.shape} | {preds.tolist()}")
        print(f"  -                       acc = {acc.shape} | {acc}")

        out_hr()
        encoded_examples: List[NEREncodedExample] = [self.val_dataset[i] for i in example_ids]
        for encoded_example in encoded_examples:
            print(f"  - encoded_example.idx       = {encoded_example.idx}")
            print(f"  - encoded_example.raw       = {encoded_example.raw}")
            print(f"  - encoded_example.encoded   = {encoded_example.encoded}")
            print(f"  - encoded_example.label_ids = {encoded_example.label_ids}")
            print()
        out_hr()
        return {
            "loss": outputs.loss, "logits": outputs.logits,
            "preds": preds, "labels": labels, "examples": encoded_examples,
        }

    def test_step(self, batch, batch_idx):
        outputs: TokenClassifierOutput = self.model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        acc = accuracy(preds, labels, ignore_index=NER_PAD_ID)
        self.log(prog_bar=False, logger=True, on_epoch=True, name="test_loss", value=outputs.loss)
        self.log(prog_bar=False, logger=True, on_epoch=True, name="test_acc", value=acc)
        return {"test_loss": outputs.loss, "test_acc": acc}

    def validation_epoch_end(
            self, outputs: List[Dict[str, torch.Tensor | List[NERRawExample]]], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        """When validation step ends, either token- or character-level predicted
        labels are aligned with the original character-level labels and then
        evaluated.
        """
        print()
        print(f"[validation_epoch_end]")
        print(f"  = outputs: {len(outputs)} * {outputs[0].keys()}")
        examples: List[NEREncodedExample] = [x for output in outputs for x in output["examples"]]
        labels: torch.Tensor = torch.cat([output["labels"] for output in outputs], dim=0)
        preds: torch.Tensor = torch.cat([output["preds"] for output in outputs], dim=0)

        print(f"  - labels({labels.shape}): {labels.tolist()}")
        print(f"  - preds ({preds.shape}): {preds.tolist()}")
        print(f"  - examples({len(examples)}):")
        for example in examples:
            example: NEREncodedExample = example
            out_hr('-')
            print(f"    - example[{example.idx}].raw.origin         = {example.raw.origin}")
            print(f"    - example[{example.idx}].raw.character_list = {example.raw.character_list}")
            print(f"    - example[{example.idx}].encoded.tokens()   = {example.encoded.tokens()}")
            print(f"    - example[{example.idx}].label_ids          = {example.label_ids}")
        out_hr('-')
        exit(1)

        original_examples = self.hparams.data[data_type]["original_examples"]
        list_of_character_preds = []
        list_of_originals = []
        label_list = self.hparams.label_list

        for i, (subword_preds, example) in enumerate(zip(list_of_subword_preds, original_examples)):
            original_sentence = example["original_sentence"]  # 안녕 하세요 ^^
            character_preds = [subword_preds[0].tolist()]  # [CLS]
            character_preds_idx = 1
            for word in original_sentence.split(" "):  # ['안녕', '하세요', '^^']
                if character_preds_idx >= self.hparams.max_seq_length - 1:
                    break
                subwords = self.tokenizer.tokenize(word)  # 안녕 -> [안, ##녕] / 하세요 -> [하, ##세요] / ^^ -> [UNK]
                if self.tokenizer.unk_token in subwords:  # 뻥튀기가 필요한 case!
                    unk_aligned_subwords = self.tokenizer_out_aligner(
                        word, subwords, strip_char
                    )  # [UNK] -> [UNK, +UNK]
                    unk_flag = False
                    for subword in unk_aligned_subwords:
                        if character_preds_idx >= self.hparams.max_seq_length - 1:
                            break
                        subword_pred = subword_preds[character_preds_idx].tolist()
                        subword_pred_label = label_list[subword_pred]
                        if subword == self.tokenizer.unk_token:
                            unk_flag = True
                            character_preds.append(subword_pred)
                            continue
                        elif subword == self.in_unk_token:
                            if subword_pred_label == "O":
                                character_preds.append(subword_pred)
                            else:
                                _, entity_category = subword_pred_label.split("-")
                                character_pred_label = "I-" + entity_category
                                character_pred = label_list.index(character_pred_label)
                                character_preds.append(character_pred)
                            continue
                        else:
                            if unk_flag:
                                character_preds_idx += 1
                                subword_pred = subword_preds[character_preds_idx].tolist()
                                character_preds.append(subword_pred)
                                unk_flag = False
                            else:
                                character_preds.append(subword_pred)
                                character_preds_idx += 1  # `+UNK`가 끝나는 시점에서도 += 1 을 해줘야 다음 label로 넘어감
                else:
                    for subword in subwords:
                        if character_preds_idx >= self.hparams.max_seq_length - 1:
                            break
                        subword = subword.replace(strip_char, "")  # xlm roberta: "▁" / others "##"
                        subword_pred = subword_preds[character_preds_idx].tolist()
                        subword_pred_label = label_list[subword_pred]
                        for i in range(0, len(subword)):  # 안, 녕
                            if i == 0:
                                character_preds.append(subword_pred)
                            else:
                                if subword_pred_label == "O":
                                    character_preds.append(subword_pred)
                                else:
                                    _, entity_category = subword_pred_label.split("-")
                                    character_pred_label = "I-" + entity_category
                                    character_pred = label_list.index(character_pred_label)
                                    character_preds.append(character_pred)
                        character_preds_idx += 1
            character_preds.append(subword_preds[-1].tolist())  # [SEP] label
            list_of_character_preds.extend(character_preds)
            original_labels = ["O"] + example["original_clean_labels"][: len(character_preds) - 2] + ["O"]
            originals = []
            for label in original_labels:
                originals.append(label_list.index(label))
            assert len(character_preds) == len(originals)
            list_of_originals.extend(originals)

        self._set_metrics_device()

        if write_predictions is True:
            self.predictions = list_of_character_preds

        for k, metric in self.metrics.items():
            metric(list_of_character_preds, list_of_originals, label_list)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    def _convert_outputs_to_preds(self, outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        logits = torch.cat([output["logits"] for output in outputs], dim=0)
        return torch.argmax(logits, axis=2)
