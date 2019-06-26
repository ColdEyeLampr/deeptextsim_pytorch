# -*- coding: utf-8 -*-
# author: Ziding Liu

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from heapq import nlargest
import argparse
from tqdm import tqdm
from model import TextSim
from data_utils import *

parser = argparse.ArgumentParser(description='Textsim RNN+Attn Version')
# learning
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for train [default: 100]')
parser.add_argument('--batch-size', type=int, default=2, help='batch size for training [default: 64]')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--l2-norm', type=float, default=1e-4, help='l2 constraint of parameters [default: 1e-4]')
parser.add_argument('--eval-interval', type=int, default=1, help='how many epochs to evaluate [default: 1]')
parser.add_argument('--early-stop', type=int, default=3,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('--save-path', type=str, default='ckpts/textsim-model', help='where to save the checkpoint')
parser.add_argument('--load-path', type=str, default=None, help='where to load the checkpoint [default: None]')
# data
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data every epoch')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# Load Data
train = pickle.load(open('/Users/zidingliu/Downloads/991/data/train1.ids'))
test = pickle.load(open('data/test.ids'))
print 'Loaded train.ids (%d)' % len(train)
print 'Loaded test.ids (%d)' % len(test)

num_steps = len(train) // args.batch_size
train = train[:num_steps * args.batch_size]
# train = sorted(train, key=lambda x: (len(x[1]), len(x[2])), reverse=True)

# Training:
classifier = TextSim(is_training=True)

print classifier
print

if use_cuda:
    print 'using gpu'
    classifier = classifier.cuda()
if args.load_path:
    try:
        classifier.load_state_dict(torch.load(args.load_path))
        print('Loaded %s' % args.load_path)
    except IOError:
        pass

criterion = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, classifier.parameters())
optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.l2_norm)


# Eval on test/val
def evaluate(predictor, batch_size):
    if use_cuda:
        predictor = predictor.cuda(0)
    # Evaluate on test:
    truth = []
    pred = []
    for i in tqdm(range(len(test) // batch_size)):
        corpus1, corpus2, labels = get_batch_data(test, i, batch_size)

        if use_cuda:
            input1 = Variable(corpus1).cuda()
            input2 = Variable(corpus2).cuda()
        else:
            input1 = Variable(corpus1)
            input2 = Variable(corpus2)

        labels = list(labels.numpy())
        ys = predictor(input1, input2)
        ys = list(ys.cpu().argmax(dim=1).numpy())

        truth.extend(labels)
        pred.extend(ys)

    acc, f1, _, _ = calc_stats(pred, truth)
    print 'Eval Acc=%.4f\nEval f1=%.4f' % (acc, f1)
    print '-' * 20
    return f1


# Training:
worse_count, acc_worse_count = 0, 0
last_loss = np.inf
f1_history = []

for epoch in range(args.epochs):
    if args.shuffle:
        np.random.shuffle(train)

    avg_loss = 0.0
    for i in range(num_steps):
        corpus1, corpus2, labels = get_batch_data(train, i, args.batch_size)

        if use_cuda:
            input1 = Variable(corpus1).cuda()
            input2 = Variable(corpus2).cuda()
            truth = Variable(labels).cuda()
        else:
            input1 = Variable(corpus1)
            input2 = Variable(corpus2)
            truth = Variable(labels)

        optimizer.zero_grad()
        pred = classifier(input1, input2)
        loss = criterion(pred, truth)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        avg_loss += loss_val

        step = (i + 1)
        if step % max(1, (num_steps // 10)) == 0:
            print 'Epoch [%d/%d], Step[%d/%d], Loss: %.3f' % (epoch + 1, args.epochs, step, num_steps, loss_val)

    avg_loss /= num_steps
    avg_loss = round(avg_loss, 3)
    print 'Epoch Avg Loss: %f' % avg_loss
    print '-' * 20

    # Compare Training loss
    if avg_loss >= last_loss:
        worse_count += 1
        acc_worse_count += 1
    else:
        # set back to zero if no consecutive worse case
        worse_count = 0
    last_loss = avg_loss

    # Adjust lr if accumulative
    if acc_worse_count > 2:
        acc_worse_count = 0
        for g in optimizer.param_groups:
            g['lr'] /= 2
            print 'Updated lr=%s' % str(g['lr'])

    # Eval
    if (epoch + 1) % args.eval_interval == 0 and epoch+1 >= 10:
        torch.save(classifier.state_dict(), args.save_path + '-epoch%03d' % (epoch + 1))
        print('Saved ' + args.save_path + '-epoch%03d' % (epoch + 1))
        predictor = TextSim(is_training=False)
        predictor.load_state_dict(classifier.state_dict())
        f1_score = evaluate(predictor, 50)
        f1_history.append((epoch + 1, f1_score))

    # Early stop condition
    if worse_count == args.early_stop:
        torch.save(classifier.state_dict(), args.save_path + '-final')
        print('Saved ' + args.save_path + '-final')
        predictor = TextSim(is_training=False)
        predictor.load_state_dict(classifier.state_dict())
        f1_score = evaluate(predictor, 50)
        f1_history.append((epoch + 1, f1_score))
        break

best = nlargest(1, f1_history, key=lambda x: x[1])[0]
print 'Best f1 ckpt: epoch%03d %.4f' % (best[0], best[1])
