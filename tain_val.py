from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import importlib
import os

from features_extract import visual_extract, load_model, texture_extract
from multi_fusion import FusionNet
from option import Options


input_size = 32768
h1 = 4096
h2 = 1024
num_classes = 10
num_epochs = 100
batch_size = 32
learning_rate = 0.0001

best_pred = 100.0
errlist_train = []
errlist_val = []

class Net(nn.Module):
    def __init__(self, input_size, h1, h2, num_classes, drop = 0.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_classes)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        out1 = self.fc1(self.drop(x))
        out1 = self.relu(out1)

        out2 = self.fc2(self.drop(out1))
        out2 = self.relu(out2)

        out3 = self.fc3(out2)
        out = self.softmax(out3)

        return out

def main():
    global best_pred, errlist_train, errlist_val
    args = Options().parse()

    dataset = importlib.import_module('model.' + args.dataset)
    Dataloader = dataset.Dataloader
    train_loader, test_loader = Dataloader().getloader()

    net = Net(input_size, h1, h2, num_classes, drop=0.5)
    net.cuda()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    def train(epoch):
        net.train()
        global best_pred, acc_train
        train_loss, correct, total = 0, 0, 0
        tbar = tqdm(train_loader, desc='\r')

        print('\n=>Epochs %i, previous best = %.4f' % (epoch, best_pred))

        for i, data in enumerate(tbar):
            train_x_inx, train_Y_inx = data

            trainX = Variable(train_x_inx.cuda())
            trainY = Variable(train_Y_inx.cuda())

            optimizer.zero_grad()
            out = net(trainX)
            loss = criterion(out, trainY)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            pred = out.data.max(1)[1]
            correct += pred.eq(trainY.data).cpu().sum()

            # _, predicted = torch.max(out, 1)
            # correct += (predicted == trainY).sum().item()
            total += trainY.size(0)
            acc = 100.0 * correct / total
            tbar.set_description('\rLoss: %.4f | Acc: %.4f%% (%d/%d)' % (train_loss / (i + 1), acc, correct, total))

        acc_train += [acc]

    def test(epoch):
        net.eval()
        global best_pred, acc_train, acc_val
        test_loss, correct, total = 0, 0, 0
        tbar = tqdm(test_loader, desc='\r')

        for batch_idx, (test_x_inx, test_Y_inx) in enumerate(tbar):
            testX = Variable(test_x_inx.cuda())
            testY = Variable(test_Y_inx.cuda())

            with torch.no_grad():
                output = net(testX)
                val_loss = criterion(output, testY)

                test_loss += val_loss.data.item()
                pred = output.data.max(1)[1]
                correct += pred.eq(testY.data).cpu().sum().item()

                # _, predicted = torch.max(output, 1)
                # correct += (predicted == testY).sum().item()
                total += testY.size(0)

            acc = 100.0 * correct / total
            tbar.set_description('Loss: %.4f | Acc: %.4f%% (%d/%d)' % (test_loss / (batch_idx + 1), acc, correct, total))

        acc_val += [acc]
        if acc < best_pred:
            best_pred = acc

        if epoch % 10 == 0:
            torch.save(net.state_dict(), 'model/VT_%s.pth' % epoch)

    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test(epoch)

    plt.clf()
    plt.xlabel('Epoches: ')
    plt.ylabel('Accuracy: %')
    plt.plot(acc_train, label='train')
    plt.plot(acc_val, label='val')
    plt.savefig('train_val.jpg')


if __name__ == '__main__':
    main()
