"""
### 참고 자료

##### 1. 논문
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

##### 2. 코드
   - [Text classification examples (transformers)](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)
   - [Image classification examples (lightning)](https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/image_classifier)
   - [Finetuning (KoELECTRA)](https://github.com/monologg/KoELECTRA/tree/master/finetune)
   - [KE-T5 Downstreams](https://github.com/AIRC-KETI/ke-t5-downstreams)
   - [Process (datasets)](https://huggingface.co/docs/datasets/process)
"""
from __future__ import annotations

from abc import ABC

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoConfig, PreTrainedModel, AutoModelForSequenceClassification, T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput

morp_tags_all = [
    "/NNG", "/NNP", "/NNB", "/NP", "/NR", "/VV", "/VA", "/VX", "/VCP", "/VCN",  # 10
    "/IC", "/JKS", "/JKC", "/JKG", "/JKO", "/JKB", "/JKV", "/JKQ", "/JX", "/JC",  # 20
    "/EP", "/EF", "/EC", "/ETN", "/ETM", "/XPN", "/XSN", "/XSV", "/XSA", "/XR",  # 30
    "/SF", "/SP", "/SS", "/SE", "/SO", "/SW", "/SL", "/SH", "/SN", "/MM",  # 40
    "/MAG", "/MAJ",  # 42
]

morp_tags_mid = [
    "/NN", "/NP", "/NR", "/VV", "/VA", "/VX", "/VC", "/XPN", "/XS", "/XR",  # 10
    "/MM", "/MA", "/IC", "/JK", "/JX", "/JC", "/EP", "/EF", "/EC", "/ET",  # 20
    "/SF", "/SP", "/SS", "/SE", "/SO", "/SW", "/SL", "/SH", "/SN",  # 29
]

morp_tags_big = [
    "/N", "/V", "/X", "/M", "/IC", "/J", "/E", "/S",  # 8
]

additive_tokens_for_morp_tag = {
    "all": morp_tags_all,
    "mid": morp_tags_mid,
    "big": morp_tags_big,
}


class SimplePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.d_model, config.d_model)
        self.dense2 = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask=None):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense1(first_token_tensor)
        pooled_output = nn.functional.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense2(pooled_output)

        return pooled_output


class BertHeadModel(PreTrainedModel, ABC):
    """
    Head for sentence-level classification or regression tasks using BERT-like model.
    - Refer to `transformers.models.big_bird.modeling_big_bird.BigBirdClassificationHead`
    - Refer to https://github.com/chrisjihee/DeepKorean/blob/master/src/train.py
    """

    def __init__(self, state, tokenizer):
        config = AutoConfig.from_pretrained(state.pretrained.path, num_labels=state.num_classes)
        super().__init__(config)
        self.model = AutoModelForSequenceClassification.from_pretrained(state.pretrained.path, config=config)
        if state.pretrained.additive_tokens:
            # https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512
            # https://github.com/huggingface/transformers/issues/1413#issuecomment-901553079
            self.model.base_model.resize_token_embeddings(len(tokenizer))
            with torch.no_grad():
                self.model.base_model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([self.model.base_model.config.hidden_size])

    def forward(self, **x):
        return self.model(**x)


class T5HeadModel(T5EncoderModel, ABC):
    """
    Head for sentence-level classification or regression tasks using T5-like model.
    - Refer to `transformers.models.t5.modeling_t5.T5EncoderModel`
    - Refer to https://github.com/AIRC-KETI/ke-t5-downstreams/blob/main/ke_t5/models/models.py
    """

    def __init__(self, state):
        config = AutoConfig.from_pretrained(state.pretrained.path, num_labels=state.num_classes)
        super().__init__(config)
        self.pooler = SimplePooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
