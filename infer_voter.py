# -*- coding: utf-8 -*-
# author: Ziding Liu

import sys
import cPickle as pickle
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
from model import TextSim


class Voter:
    def __init__(self, k):
        self.k = k
        self.char2ids = []
        self.classifiers = []

        for i in range(k):
            dictpath = 'utils/' + str(i) + '/char2id'
            char2id = pickle.load(open(dictpath))
            self.char2ids.append(char2id)

            modelpath = 'ckpts/textsim-model-k' + str(i)
            classifier = TextSim(is_training=False, vocab_size=len(char2id) + 1)
            classifier.load_state_dict(torch.load(modelpath, map_location=lambda storage, loc: storage))
            self.classifiers.append(classifier)

    def sent2ids(self, sent, model_id):
        char2id = self.char2ids[model_id]
        unk = len(char2id) - 1
        eos = len(char2id)
        ids = [char2id[c] if c in char2id else unk for c in sent] + [eos]
        return Variable(torch.LongTensor([ids]))

    def vote(self, sen1, sen2):
        ys = []
        for i in range(self.k):
            input1 = Variable(self.sent2ids(sen1, i))
            input2 = Variable(self.sent2ids(sen2, i))
            classifier = self.classifiers[i]
            y = classifier(input1, input2).view(-1).cpu().argmax().item()
            ys.append(y)
        return np.argmax(np.bincount(ys))


def process(inpath, outpath):
    voter = Voter(5)

    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2 = line.decode('utf-8').strip().split('\t')
            y = voter.vote(sen1, sen2)
            if y == 1:
                fout.write(lineno.encode('utf-8') + '\t1\n')
            else:
                fout.write(lineno.encode('utf-8') + '\t0\n')


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
