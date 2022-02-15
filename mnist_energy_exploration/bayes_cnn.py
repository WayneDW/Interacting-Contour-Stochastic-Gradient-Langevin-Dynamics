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

from models.model_zoo import CNN, CNN5


def main():
    parser = argparse.ArgumentParser(description='Grid search')
    parser.add_argument('-c', default='csghmc', help='classifier: csgmcmc or sgmcmc')

    # numper of optimization/ sampling epochs
    parser.add_argument('-sn', default=1000, type=int, help='Sampling Epochs')
    parser.add_argument('-classes', default=2, type=int, help='classify the first N classes')
    parser.add_argument('-filters', default=16, type=int, help='number of filters')
    parser.add_argument('-hidden', default=10, type=int, help='number of hidden nodes')
    parser.add_argument('-batch', default=5000, type=int, help='Training batch size')
    parser.add_argument('-N', default=50000, type=int, help='Total training data')
    parser.add_argument('-wdecay', default=25, type=float, help='Samling weight decay')
    parser.add_argument('-lr', default=1e-7, type=float, help='Sampling learning rate')
    parser.add_argument('-warm', default=0.5, type=float, help='warm up period with large learning rates')
    parser.add_argument('-stepsize', default=0.01, type=float, help='stepsize for stochastic approximation')
    parser.add_argument('-part', default=100000, type=int, help='The number of partitions')
    parser.add_argument('-div', default=10, type=float, help='Divide energy: divisor to calculate partition index')
    parser.add_argument('-zeta', default=1e5, type=float, help='Adaptive amplifier')
    parser.add_argument('-T', default=1.0, type=float, help='Tempreture')
    parser.add_argument('-ifprint', default=0, type=float, help='if print subprocess')

    # other settings
    parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
    parser.add_argument('-gpu', default=0, type=int, help='Default GPU')

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

    pars.lr =  pars.lr * (10 / pars.classes)
    pars.wdecay = pars.wdecay / (10 / pars.classes)
    print('adjust the learning rate {:.3e} weight decay {:.3e}'.format(pars.lr, pars.wdecay)) 

    """ Step 1: Preprocessing """
    if not torch.cuda.is_available():
        exit("CUDA does not exist!!!")

    if pars.classes == 2:
        net = CNN(pars.classes, pars.filters, pars.hidden).cuda()
    else:
        net = CNN5(pars.classes, pars.filters, pars.hidden).cuda()
    
    for i, param in enumerate(net.parameters()):
        print(param.shape)
    
    """ Step 2: Load Data """
    train_loader, test_loader = loader(pars.batch, pars.batch, pars)

    """ Step 3: Bayesian Sampling """
    sgmcmc(net, train_loader, test_loader, pars)
    
if __name__ == "__main__":
    main()
