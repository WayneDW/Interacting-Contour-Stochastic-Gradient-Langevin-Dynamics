#!/usr/bin/python
import math
import copy
import sys
import os
import timeit
import csv
import argparse
import pickle
import random
from random import shuffle

from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np

## import pytorch modules
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets

## Import helper functions
from tools import BayesEval
from sgmcmc import Sampler

CUDA_EXISTS = torch.cuda.is_available()


def sgmcmc(net, train_loader, test_loader, pars):
    start = timeit.default_timer()
    samplers, BMA = [], BayesEval(pars.classes)
    criterion = nn.CrossEntropyLoss()
    sampler = Sampler(net, pars)

    import_w = 0

    for epoch in range(pars.sn):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
            filters = (labels < pars.classes)
            images, labels = images[filters], labels[filters]
            net.train()
            sampler.net.zero_grad()
            """ change mean loss to sum loss to adapt to Bayesian settings """
            loss = criterion(sampler.net(images), labels) * pars.classes * 6000
            loss.backward()
            import_w = sampler.cstep(images, labels, pars.stepsize, loss.item())
            print('Epoch {} Iter {} subLoss {:0.1f} gradient multilier {:.2f} importance weight {:.2f}'.format(epoch, i, loss, sampler.gmul, sampler.import_weight))

        """ apply weight 1 using SGHMC and importance weights using Contour SGHMC """
        BMA.eval(net, train_loader, test_loader, criterion, pars.classes, weight=import_w, bma=False)

        pdfEnergy = sampler.G.cpu().numpy()
        print('Epoch {} Acc: {:0.2f} BMA: {:0.2f} lr: {:.2E} T: {:.2E}  Weight {:.3f} gradient multilier {:.2f} Pidx {} train Loss: {:0.1f} test Loss: {:0.1f}'.format(\
            epoch, BMA.cur_acc,  BMA.bma_acc, sampler.lr,  sampler.T, import_w, sampler.gmul, sampler.J, BMA.train_loss, BMA.test_loss))
       
    end = timeit.default_timer()
    print("Sampling Time used: {:0.1f}".format(end - start))

    print('Grad mul')
    multipliers = 1 + sampler.zeta * sampler.T * (np.log(pdfEnergy[1:]) - np.log(pdfEnergy[:-1])) / sampler.div
