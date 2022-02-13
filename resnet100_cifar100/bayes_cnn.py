"""
Code for Interacting-Contour-Stochastic-Gradient-Langevin-Dynamics
(c) Wei Deng @ Purdue University
Nov 8, 2021
"""

#!/usr/bin/python

import math
import copy
import sys
import os
import timeit
import csv
import argparse
from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np
import random
import pickle
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

from tools import loader
from trainer import sgmcmc

import models.cifar as cifar_models

'''
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
'''


def main():
    parser = argparse.ArgumentParser(description='Grid search')
    parser.add_argument('-c', default='csghmc', help='classifier: csgmcmc or sgmcmc')
    parser.add_argument('-data', default='cifar100', dest='data', help='CIFAR10/ CIFAR100')
    parser.add_argument('-model', default='resnet', type=str, help='resnet20 model')
    parser.add_argument('-depth', type=int, default=20, help='Model depth.')

    # numper of optimization/ sampling epochs
    parser.add_argument('-sn', default=500, type=int, help='Sampling Epochs')
    parser.add_argument('-chains', default=4, type=int, help='Total number of chains')
    parser.add_argument('-batch', default=256, type=int, help='Training batch size')
    parser.add_argument('-N', default=50000, type=int, help='Total training data')
    parser.add_argument('-wdecay', default=25, type=float, help='Samling weight decay')
    parser.add_argument('-lr', default=2e-6, type=float, help='Sampling learning rate')
    parser.add_argument('-lowgrad', default=-5, type=float, help='lower bound for gradient multiplier for stability')
    parser.add_argument('-upgrad', default=5, type=float, help='upper bound for gradient multiplier for stability')
    parser.add_argument('-momentum', default=0.9, type=float, help='Sampling momentum learning rate')
    parser.add_argument('-warm', default=0.5, type=float, help='warm up period with large learning rates')
    parser.add_argument('-burn', default=0.6, type=float, help='burn in iterations for sampling (sn * burn)')
    parser.add_argument('-period', default=2, type=int, help='update every [p.] epochs. NO VR by default')

    parser.add_argument('-stepsize', default=0, type=float, help='stepsize for stochastic approximation')
    parser.add_argument('-part', default=0, type=int, help='The number of partitions')
    parser.add_argument('-div', default=0, type=float, help='Divide energy: divisor to calculate partition index')
    parser.add_argument('-bias', default=0, type=float, help='Minimum energy: Bias to calculate partition index')
    parser.add_argument('-zeta', default=0, type=float, help='Adaptive amplifier')

    # Important tuning hyperparameters
    parser.add_argument('-T', default=0.0003, type=float, help='Tempreture')
    parser.add_argument('-LRanneal', default=0.2, type=float, help='lr decay factor')

    """ hyperparameter for the baseline: replica exchange """
    parser.add_argument('-scalar_T', default=3, type=float, help='Tempreture ladder scalar')
    parser.add_argument('-scalar_lr', default=1.0, type=float, help='Learning rate ladder scalar')
    parser.add_argument('-correction', default=0, type=int, help='correction')

    # other settings
    parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
    parser.add_argument('-gpu', default=0, type=int, help='Default GPU')
    parser.add_argument('-cycle', default=1, type=int, help='Number of cycles')

    pars = parser.parse_args()
    """ Step 0: Numpy printing setup and set GPU and Seeds """
    print(pars)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    try:
        torch.cuda.set_device(pars.gpu)
    except: # in case the device has only one GPU
        torch.cuda.set_device(0) 
    torch.manual_seed(pars.seed)
    torch.cuda.manual_seed(pars.seed)
    np.random.seed(pars.seed)
    random.seed(pars.seed)
    torch.backends.cudnn.deterministic=True

    """ Step 1: Preprocessing """
    if not torch.cuda.is_available():
        exit("CUDA does not exist!!!")

    nets = []
    for idx in range(pars.chains):
        if pars.model.startswith('resnet'):
            no_c = 10
            if pars.data == 'cifar100':
                no_c = 100
            for idx in range(pars.chains):
                net = cifar_models.__dict__['resnet'](num_classes=no_c, depth=pars.depth).cuda()
        elif pars.model == 'wrn':
            net = cifar_models.__dict__['wrn'](num_classes=100, depth=16, widen_factor=8, dropRate=0).cuda()
        elif pars.model == 'wrn28':
            net = cifar_models.__dict__['wrn'](num_classes=100, depth=28, widen_factor=10, dropRate=0).cuda()
        else:
            print('Unknown Model structure')
        nets.append(pickle.loads(pickle.dumps(net)))

    
    """ Step 2: Load Data """
    train_loader, test_loader = loader(pars.batch, pars.batch, pars)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    notcifar = datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform_test)
    extra_loader = data.DataLoader(notcifar, batch_size=pars.batch, shuffle=False, num_workers=0)

    """ Step 3: Bayesian Sampling """
    sgmcmc(nets, train_loader, test_loader, extra_loader, pars)
    
if __name__ == "__main__":
    main()
