# -*- coding: utf-8 -*-
# author: Ziding Liu

import torch.nn as nn
from torch.nn import functional as F
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
                              dropout=self.dropout_prob, bidirectional=True)

        # Output Layer
        self.mlp = nn.Linear(hidden_size * 4, mlp_hidden_size)
        self.output = nn.Linear(mlp_hidden_size, num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.mlp.weight)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, input1, input2):
        _attn_mask1 = get_attn_padding_mask(input1, input2)
        _attn_mask2 = get_attn_padding_mask(input2, input1)

        # Embedding Layer:
        # [batch_size x seq_len] => [batch_size x seq_len x hidden_size]
        embed1 = self.embed(input1)
        embed2 = self.embed(input2)

        # Encoder Layer:
        # u: [batch_size x seq_len1 x hidden_size * 2]
        # v: [batch_size x seq_len2 x hidden_size * 2]
        u, _ = self.encoder(embed1)
        v, _ = self.encoder(embed2)

        # 2-way Attention Layer: softmax(QK)V

        attn1 = torch.bmm(u, v.transpose(1, 2))
        attn1.data.masked_fill_(_attn_mask1, -float('inf'))
        attn1 = F.softmax(attn1, dim=2)

        attn2 = torch.bmm(v, u.transpose(1, 2))
        attn2.data.masked_fill_(_attn_mask2, -float('inf'))
        attn2 = F.softmax(attn2, dim=2)

        va = torch.bmm(attn1, v)  # [batch_size x seq_len1 x hidden_size * 2]
        ua = torch.bmm(attn2, u)  # [batch_size x seq_len2 x hidden_size * 2]

        s = torch.cat([torch.max(u, dim=1)[0], torch.max(ua, dim=1)[0],
                       torch.max(v, dim=1)[0], torch.max(va, dim=1)[0]], dim=1)

        # Output Layer
        out = self.mlp(s)
        pred = self.output(F.dropout(F.relu(out), p=self.dropout_prob, training=self.is_training))

        if self.ret_attn:
            return pred, attn1, attn2
        else:
            return pred
