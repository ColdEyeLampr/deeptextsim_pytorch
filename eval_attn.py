# -*- coding: utf-8 -*-
# author: Ziding Liu

import sys
from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as ticker

from model import TextSim
from data_utils import *

myfont = matplotlib.font_manager.FontProperties(fname='/Library/Fonts/Songti.ttc')


def sent2ids(sent):
    ids = [char2id[c] if c in char2id else unk for c in sent] + [eos]
    return Variable(torch.LongTensor([ids]))


def vis(attn, xseq, yseq):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(attn, cmap='bone')

    # Set up axes
    ax.set_xticklabels(xseq, fontsize=10, fontproperties=myfont)
    ax.set_yticklabels(yseq, fontsize=10, fontproperties=myfont)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def process(inpath):
    classifier = TextSim(is_training=False, ret_attn=True)
    classifier.load_state_dict(torch.load('textsim-model-epoch029', map_location=lambda storage, loc: storage))

    fin = open(inpath, 'r').readlines()

    for line in fin[:100]:
        lineno, sen1, sen2, sim = line.decode('utf-8').strip().split('\t')
        input1 = Variable(sent2ids(sen1))
        input2 = Variable(sent2ids(sen2))
        pred, attn = classifier(input1, input2)
        y = pred.view(-1).cpu().argmax().item()
        if int(sim) == 1 and y == 1:
            attn = np.array(attn.data).squeeze(axis=0)
            yseq = [''] + [u'<eos>' if c == 1692 else id2char[c] for c in np.array(input1.data).squeeze(axis=0)]
            xseq = [''] + [u'<eos>' if c == 1692 else id2char[c] for c in np.array(input2.data).squeeze(axis=0)]
            vis(attn, xseq, yseq)
            print sim, y


if __name__ == '__main__':
    process('991_test_raw.txt')

