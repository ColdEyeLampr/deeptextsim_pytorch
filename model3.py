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
                 mlp_hidden_size=1024, dropout_prob=0.5, num_classes=2, ret_attn=False):
        super(TextSim, self).__init__()
        self.dropout_prob = dropout_prob if is_training else 0.0
        self.is_training = is_training
        self.ret_attn = ret_attn

        # Embedding Layer
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        # Encoder Layer
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True,
                              dropout=self.dropout_prob, bidirectional=True)

        # Attention Layer
        self.attn_W = nn.Linear(hidden_size * 6, 1, bias=False)

        # Feature Extraction Layer
        self.fe = nn.GRU(hidden_size * 4, hidden_size * 2, num_layers, batch_first=True,
                         dropout=self.dropout_prob, bidirectional=True)

        # Output Layer
        self.mlp = nn.Linear(hidden_size * 8, mlp_hidden_size)
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

        # Encoder Layer:
        # q: [batch_size x T x hidden_size * 2]
        # k: [batch_size x J x hidden_size * 2]
        q, _ = self.encoder(embed1)
        k, _ = self.encoder(embed2)

        # Attention Layer:
        # tri-linear attn = W[q;k;q*k]
        q_ex = q.unsqueeze(2).expand(-1, -1, k.size(1), -1)  # [batch_size x T x J x hidden_size * 2]
        k_ex = k.unsqueeze(1).expand(-1, q.size(1), -1, -1)  # [batch_size x T x J x hidden_size * 2]
        qk_concat = torch.cat([q_ex, k_ex, torch.mul(q_ex, k_ex)], dim=3)  # [batch_size x T x J x hidden_size * 6]

        attn = self.attn_W(qk_concat).squeeze(dim=-1)  # [batch_size x T x J]
        attn.data.masked_fill_(_attn_mask, -float('inf'))
        attn = F.softmax(attn, dim=2)
        v = torch.bmm(attn, k)  # [batch_size x T x hidden_size * 2]
        # TODO: k2q attn

        # Feature Extraction Layer:
        # concat query & value => GRU fe => concat fe & input
        s = torch.cat([q, v], dim=2)    # [batch_size x T x hidden_size * 4]
        m, _ = self.fe(s)   # [batch_size x T x hidden_size * 4]
        o = torch.cat([s, m], dim=2)    # [batch_size x T x hidden_size * 8]

        # Output Layer
        out = self.mlp(torch.max(o, dim=1)[0])
        pred = self.output(F.dropout(F.relu(out), p=self.dropout_prob, training=self.is_training))

        if self.ret_attn:
            return pred, attn
        else:
            return pred
