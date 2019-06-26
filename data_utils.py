# -*- coding: utf-8 -*-
# author: Ziding Liu

import torch
import cPickle as pickle

char2id = pickle.load(open('utils/char2id'))
id2char = pickle.load(open('utils/id2char'))
vocab_size = len(char2id) + 1
unk = len(char2id) - 1
assert id2char[unk] == u'_UNK'

eos = len(char2id)


def pad_seq_len(corpus, seq_len, pad=0):
    res = []
    for text in corpus:
        if len(text) != seq_len:
            res.append(text + [eos] + [pad] * (seq_len - len(text)))
        else:
            res.append(text + [eos])
    return res


def get_batch_data(data, i, batch_size):
    batch_data = data[i * batch_size: (i + 1) * batch_size]
    corpus1, corpus2 = [], []
    seq_len1, seq_len2 = 0, 0
    labels = torch.LongTensor(batch_size).fill_(0)
    for j in range(batch_size):
        _, text1, text2, sim = batch_data[j]
        corpus1.append(text1)
        if len(text1) > seq_len1:
            seq_len1 = len(text1)
        corpus2.append(text2)
        if len(text2) > seq_len2:
            seq_len2 = len(text2)
        labels[j] = sim

    corpus1 = pad_seq_len(corpus1, seq_len1)
    corpus2 = pad_seq_len(corpus2, seq_len2)

    return torch.LongTensor(corpus1), torch.LongTensor(corpus2), labels


def calc_stats(truth, pred):
    assert len(truth) == len(pred)
    tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
    for i in range(len(truth)):
        if pred[i] == 1 and truth[i] == 1:
            tp += 1
        if pred[i] == 1 and truth[i] == 0:
            fp += 1
        if pred[i] == 0 and truth[i] == 1:
            fn += 1
        if pred[i] == 0 and truth[i] == 0:
            tn += 1
    acc = (tp + tn) / (tp + fp + fn + tn)
    prec = tp / (tp + fp) if (tp + fp) != 0.0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) != 0.0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0.0 else 0.0
    return acc, f1, prec, rec