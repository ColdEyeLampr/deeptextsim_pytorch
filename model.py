# -*- coding: utf-8 -*-
# author: Ziding Liu

import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from data_utils import *


def get_attn_padding_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask


class TextSim(nn.Module):

    def __init__(self, is_training, vocab_size=vocab_size, hidden_size=128, num_layers=2,
                 mlp_hidden_size=512, dropout_prob=0.5, num_classes=2, ret_attn=False):
        super(TextSim, self).__init__()
        self.dropout_prob = dropout_prob if is_training else 0.0
        self.is_training = is_training
        self.ret_attn = ret_attn

        # Embedding Layer
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        # Encoder Layer
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True,
                              dropout=self.dropout_prob, bidirectional=False)

        # Output Layer
        self.mlp = nn.Linear(hidden_size * 4, mlp_hidden_size)
        self.output = nn.Linear(mlp_hidden_size, num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.mlp.weight)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, input1, input2):
        _attn_mask = get_attn_padding_mask(input1, input2)

        # Embedding Layer:
        # [batch_size x seq_len] => [batch_size x seq_len x hidden_size]
        embed1 = self.embed(input1)
        embed2 = self.embed(input2)

        # padding:
        pack_in = nn.utils.rnn.pack_padded_sequence(embed1, [15, 9, 7], batch_first=True)
        pack_out, _ = self.encoder(pack_in)
        qq, _ = nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)

        # Encoder Layer:
        # q: [batch_size x seq_len1 x hidden_size * 2]
        # k: [batch_size x seq_len2 x hidden_size * 2]
        q, _ = self.encoder(embed1)
        k, _ = self.encoder(embed2)

        # Attention Layer: softmax(QK)V
        attn = torch.bmm(q, k.transpose(1, 2))
        attn.data.masked_fill_(_attn_mask, -float('inf'))
        attn = F.softmax(attn, dim=2)
        v = torch.bmm(attn, k)  # [batch_size x seq_len1 x hidden_size * 2]

        s = torch.cat([torch.max(q, dim=1)[0], torch.max(v, dim=1)[0]], dim=1)

        # Output Layer
        out = self.mlp(s)
        pred = self.output(F.dropout(F.relu(out), p=self.dropout_prob, training=self.is_training))

        if self.ret_attn:
            return pred, attn
        else:
            return pred


if __name__ == '__main__':
    train = pickle.load(open('data/test.ids'))

    m = TextSim(is_training=False)

    corpus1, corpus2, labels = get_batch_data(train, 0, 3)
    pred = m(corpus1, corpus2)
    print pred
