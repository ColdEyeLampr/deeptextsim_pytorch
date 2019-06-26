# -*- coding: utf-8 -*-
# author: Ziding Liu

import sys
from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
from model import TextSim
from data_utils import *


def sent2ids(sent):
    ids = [char2id[c] if c in char2id else unk for c in sent] + [eos]
    return Variable(torch.LongTensor([ids]))


def process(inpath, outpath):
    classifier = TextSim(is_training=False)
    classifier.load_state_dict(torch.load('textsim-model-final', map_location=lambda storage, loc: storage))

    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2 = line.decode('utf-8').strip().split('\t')
            input1 = Variable(sent2ids(sen1))
            input2 = Variable(sent2ids(sen2))
            y = classifier(input1, input2)
            y = y.view(-1).cpu().argmax().item()
            if y == 1:
                fout.write(lineno.encode('utf-8') + '\t1\n')
            else:
                fout.write(lineno.encode('utf-8') + '\t0\n')


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])

