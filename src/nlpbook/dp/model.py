from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nlpbook.dp.corpus import DPCorpus
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class ModelForDependencyParsing(nn.Module):
    def __init__(self,
                 corpus: DPCorpus,
                 pretrained_model: PreTrainedModel):
        super().__init__()
        self.encoder_layers = 1  # Number of layers of encoder
        self.decoder_layers = 1  # Number of layers of decoder
        self.hidden_size = 768  # Number of hidden units in LSTM
        self.arc_space = 512  # Dimension of tag space
        self.type_space = 256  # Dimension of tag space
        self.pos_dim = 256  # Dimension of pos embedding
        self.input_size = pretrained_model.config.hidden_size

        self.n_pos_labels = len(corpus.get_pos_labels())
        self.n_dp_labels = len(corpus.get_dep_labels())
        self.pos_embedding = nn.Embedding(self.n_pos_labels + 1, self.pos_dim)

        enc_dim = self.input_size * 2
        enc_dim += self.pos_dim

        self.encoder = nn.LSTM(
            enc_dim, self.hidden_size, self.encoder_layers,
            batch_first=True, dropout=0.33,
            bidirectional=True,
        )
        self.decoder = nn.LSTM(
            self.hidden_size, self.hidden_size, self.decoder_layers,
            batch_first=True, dropout=0.33
        )
        self.dropout = nn.Dropout(p=0.33)  # TODO: check if this is correct
        # self.dropout = nn.Dropout2d(p=0.33)  # TODO: check if this is correct
        # """UserWarning: dropout2d: Received a 3D input to dropout2d and assuming that channel-wise 1D dropout behavior is desired - input is interpreted as shape (N, C, L), where C is the channel dim. This behavior will change in a future release to interpret the input as one without a batch dimension, i.e. shape (C, H, W). To maintain the 1D channel-wise dropout behavior, please switch to using dropout1d instead."""
        # self.dropout = nn.Dropout1d(p=0.33)
        self.pretrained_model = pretrained_model

        self.src_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.hx_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.arc_c = nn.Linear(self.hidden_size * 2, self.arc_space)
        self.type_c = nn.Linear(self.hidden_size * 2, self.type_space)
        self.arc_h = nn.Linear(self.hidden_size, self.arc_space)
        self.type_h = nn.Linear(self.hidden_size, self.type_space)

        self.attention = BiAttention(self.arc_space, self.arc_space, 1)
        self.bilinear = BiLinear(self.type_space, self.type_space, self.n_dp_labels)

    def forward(
            self,
            bpe_head_mask: torch.Tensor,
            bpe_tail_mask: torch.Tensor,
            pos_ids: torch.Tensor,
            head_ids: torch.Tensor,
            max_word_length: int,
            mask_e: torch.Tensor,
            mask_d: torch.Tensor,
            batch_index: torch.Tensor,
            is_training: bool = True,
            **inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.pretrained_model(**inputs)
        outputs: torch.Tensor = outputs[0]
        outputs, sent_len = self.resize_outputs(outputs, bpe_head_mask, bpe_tail_mask, max_word_length)

        pos_outputs = self.pos_embedding(pos_ids)
        pos_outputs = self.dropout(pos_outputs)
        outputs: torch.Tensor = torch.cat([outputs, pos_outputs], dim=2)

        # encoder
        packed_outputs = pack_padded_sequence(outputs, sent_len, batch_first=True, enforce_sorted=False)
        encoder_outputs, hn = self.encoder(packed_outputs)
        encoder_outputs, outputs_len = pad_packed_sequence(encoder_outputs, batch_first=True)
        encoder_outputs = self.dropout(encoder_outputs.transpose(1, 2)).transpose(1, 2)  # apply dropout for last layer
        hn = self._transform_decoder_init_state(hn)

        # decoder
        src_encoding = F.elu(self.src_dense(encoder_outputs[:, 1:]))
        sent_len = [i - 1 for i in sent_len]
        packed_outputs = pack_padded_sequence(src_encoding, sent_len, batch_first=True, enforce_sorted=False)
        decoder_outputs, _ = self.decoder(packed_outputs, hn)
        decoder_outputs, outputs_len = pad_packed_sequence(decoder_outputs, batch_first=True)
        decoder_outputs = self.dropout(decoder_outputs.transpose(1, 2)).transpose(1, 2)  # apply dropout for last layer

        # compute output for arc and type
        arc_c = F.elu(self.arc_c(encoder_outputs))
        type_c = F.elu(self.type_c(encoder_outputs))

        arc_h = F.elu(self.arc_h(decoder_outputs))
        type_h = F.elu(self.type_h(decoder_outputs))

        out_arc = self.attention(arc_h, arc_c, mask_d=mask_d, mask_e=mask_e).squeeze(dim=1)

        # use predicted head_ids when validation step
        if not is_training:
            head_ids = torch.argmax(out_arc, dim=2)

        type_c = type_c[batch_index, head_ids.data.t()].transpose(0, 1).contiguous()
        out_type = self.bilinear(type_h, type_c)

        return out_arc, out_type

    def resize_outputs(
            self, outputs: torch.Tensor, bpe_head_mask: torch.Tensor, bpe_tail_mask: torch.Tensor, max_word_length: int
    ) -> Tuple[torch.Tensor, List]:
        """Resize output of pre-trained transformers (bsz, max_token_length, hidden_dim) to word-level outputs (bsz, max_word_length, hidden_dim*2). """
        batch_size, input_size, hidden_size = outputs.size()
        word_outputs = torch.zeros(batch_size, max_word_length + 1, hidden_size * 2).to(outputs.device)
        sent_len = list()

        for batch_id in range(batch_size):
            head_ids = [i for i, token in enumerate(bpe_head_mask[batch_id]) if token == 1]
            tail_ids = [i for i, token in enumerate(bpe_tail_mask[batch_id]) if token == 1]
            assert len(head_ids) == len(tail_ids)

            word_outputs[batch_id][0] = torch.cat(
                (outputs[batch_id][0], outputs[batch_id][0])
            )  # replace root with [CLS]
            for i, (head, tail) in enumerate(zip(head_ids, tail_ids)):
                word_outputs[batch_id][i + 1] = torch.cat((outputs[batch_id][head], outputs[batch_id][tail]))
            sent_len.append(i + 2)

        return word_outputs, sent_len

    def _transform_decoder_init_state(self, hn: torch.Tensor) -> torch.Tensor:
        hn, cn = hn
        cn = cn[-2:]  # take the last layer
        _, batch_size, hidden_size = cn.size()
        cn = cn.transpose(0, 1).contiguous()
        cn = cn.view(batch_size, 1, 2 * hidden_size).transpose(0, 1)
        cn = self.hx_dense(cn)
        if self.decoder.num_layers > 1:
            cn = torch.cat(
                [
                    cn,
                    torch.autograd.Variable(cn.data.new(self.decoder.num_layers - 1, batch_size, hidden_size).zero_()),
                ],
                dim=0,
            )
        hn = torch.tanh(cn)
        hn = (hn, cn)
        return hn


class BiAttention(nn.Module):
    def __init__(self, input_size_encoder: int, input_size_decoder: int, num_labels: int, biaffine: bool = True, **kwargs):
        super(BiAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter("U", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_d)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(
            self,
            input_d: torch.Tensor,
            input_e: torch.Tensor,
            mask_d: Optional[torch.Tensor] = None,
            mask_e: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input_d.size(0) == input_e.size(0)
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        if self.biaffine:
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))
            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output


class BiLinear(nn.Module):
    def __init__(self, left_features: int, right_features: int, out_features: int):
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left: torch.Tensor, input_right: torch.Tensor) -> torch.Tensor:
        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], "batch size of left and right inputs mis-match: (%s, %s)" % (
            left_size[:-1],
            right_size[:-1],
        )
        batch = int(np.prod(left_size[:-1]))

        input_left = input_left.contiguous().view(batch, self.left_features)
        input_right = input_right.contiguous().view(batch, self.right_features)

        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
        return output.view(left_size[:-1] + (self.out_features,))
