"""
Code for Interacting-Contour-Stochastic-Gradient-Langevin-Dynamics
(c) Anonymous authors
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

from tools import save_or_pretrain, loader
#from trainer import trainer
#from trainer_cyc import trainer
from trainer_re_cyc import trainer
#from trainer_period import trainer
#from trainer_adaptive_c import trainer
#from trainer_SWA_v2 import trainer

import models.fashion as fmnist_models
import models.cifar as cifar_models
from models.cifar import PyramidNet as PYRM

parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-aug', default=1, type=float, help='Data augmentation or not')
parser.add_argument('-split', default=2, type=int, help='Bayes avg every split epochs')
# numper of optimization/ sampling epochs
parser.add_argument('-sn', default=500, type=int, help='Sampling Epochs')
parser.add_argument('-wdecay', default=25, type=float, help='Samling weight decay')
parser.add_argument('-lr', default=2e-6, type=float, help='Sampling learning rate')
parser.add_argument('-momentum', default=0.9, type=float, help='Sampling momentum learning rate')
parser.add_argument('-burn', default=0.6, type=float, help='burn in iterations for sampling (sn * burn)')
parser.add_argument('-ifstop', default=1, type=int, help='stop iteration if acc is too low')

# Parallel Tempering hyperparameters
parser.add_argument('-chains', default=2, type=int, help='Total number of chains')
parser.add_argument('-var_reduce', default=1, type=int, help='n>0 means update variance reduction every n epochs; n divides 10')
parser.add_argument('-period', default=2, type=int, help='estimate adaptive variance every [period] epochs')
parser.add_argument('-T', default=0.001, type=float, help='Inverse temperature for high temperature chain')
parser.add_argument('-T_scale', default=1.0, type=float, help='Uncertainty calibration')
parser.add_argument('-Tgap', default=0.2, type=float, help='Temperature gap between chains')
parser.add_argument('-LRgap', default=0.66, type=float, help='Learning rate gap between chains')
parser.add_argument('-anneal', default=1.002, type=float, help='temperature annealing factor')
parser.add_argument('-lr_anneal', default=0.992, type=float, help='lr annealing factor')
parser.add_argument('-F_jump', default=0.9, type=float, help='F jump factor')
parser.add_argument('-cool', default=0, type=int, help='No swaps happen during the cooling time after a swap')

# other settings
parser.add_argument('-ck', default=False, type=bool, help='Check if we need overwriting check')
parser.add_argument('-data', default='cifar10', dest='data', help='MNIST/ Fashion MNIST/ CIFAR10/ CIFAR100')
#parser.add_argument('-no_c', default=100, type=int, help='number of classes')
parser.add_argument('-model', default='resnet', type=str, help='resnet / preact / WRN')
parser.add_argument('-depth', type=int, default=20, help='Model depth.')
parser.add_argument('-total', default=50000, type=int, help='Total data points')
parser.add_argument('-train', default=256, type=int, help='Training batch size')
parser.add_argument('-test', default=1000, type=int, help='Testing batch size')
parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
parser.add_argument('-gpu', default=0, type=int, help='Default GPU')
parser.add_argument('-multi', default=0, type=int, help='Multiple GPUs')
parser.add_argument('-windows', default=20, type=int, help='Moving average of corrections')
parser.add_argument('-alpha', default=0.3, type=float, help='forgetting rate')
parser.add_argument('-bias_F', default=2000, type=float, help='correction factor F')
parser.add_argument('-cycle', default=2, type=int, help='Number of cycles')
parser.add_argument('-F_stop', default=0.8, type=float, help='F decay stop station')
parser.add_argument('-repeats', default=50, type=int, help='number of samples to estimate sample std')

pars = parser.parse_args()


torch.cuda.set_device(pars.gpu)

net = cifar_models.__dict__['resnet'](num_classes=10, depth=pars.depth).cuda()

dataloader = datasets.CIFAR10

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = dataloader(root='./data/' + pars.data.upper(), train=False, download=False, transform=transform_test)
test_loader = data.DataLoader(testset, batch_size=pars.test, shuffle=False, num_workers=0)

notcifar = datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform_test)
target_loader = data.DataLoader(notcifar, batch_size=pars.test, shuffle=False, num_workers=0)


""" Step 3: Load Model """
torch.set_printoptions(precision=3)

def number_digits(x): return str(x)[:6]
softmax = nn.Softmax(dim=1)


""" Step 3.1: Ensemble outputs and then transform to prob """

Brier_seen = 0
Brier_unseen = 0
entropy_seen = 0
entropy_unseen = 0



output_ensemble_seen = []
output_ensemble_unseen = []
prob_ensemble_seen = []
prob_ensemble_unseen = []

idx = 1

sub_sn = pars.sn / pars.cycle


seeds_list = [pars.seed]

# VR-reSGHMC
seeds_list = [4114, 24072, 54346, 92678, 99759]

# M-SGD

seeds_list = [5122, 65045, 66219, 69629, 92098]

# CSGHMC
#seeds_list = [11576, 14939]

# T  = 1
#seeds_list = [43204, 60252, 10993, 95810, 44456]

# T = 0.01
#seeds_list = [11766, 33390, 73113, 74527] 

# T = 0.03
#seeds_list = [34712, 60003, 8574, 96052]
#drwxrwxr-x 2 deng106 deng106  4096 Sep 16 23:19 cifar10_resnet20_batch_256_chain_2_T_0.03_VR_1_p_2_burn_0.6_seed_34712
#drwxrwxr-x 2 deng106 deng106  4096 Sep 16 23:19 cifar10_resnet20_batch_256_chain_2_T_0.03_VR_1_p_2_burn_0.6_seed_60003
#drwxrwxr-x 2 deng106 deng106  4096 Sep 16 23:19 cifar10_resnet20_batch_256_chain_2_T_0.03_VR_1_p_2_burn_0.6_seed_8574
#drwxrwxr-x 2 deng106 deng106  4096 Sep 16 23:19 cifar10_resnet20_batch_256_chain_2_T_0.03_VR_1_p_2_burn_0.6_seed_96052


#seeds_list = [12154, 27318, 30595, 4080]
#seeds_list = [38432, 6167, 87392]

# ensemble prob gives way better uniform predictions than taking ensemble prediction and softmax


for seed in seeds_list:
    if pars.chains == 2 and pars.var_reduce == 0:
        DIR = 'output/test_snapshot/' + pars.data + '_' + pars.model + str(pars.depth) + '_batch_' + str(pars.train) + '_chain_' + str(pars.chains) + '_T_' + str(pars.T) + '_VR_' + str(pars.var_reduce) + '_p_' + str(pars.period) + '_burn_' + str(pars.burn) + '_cycle_' + str(pars.cycle) + '_seed_' + str(seed)
    elif pars.chains == 2:
        DIR = 'output/test_snapshot/' + pars.data + '_' + pars.model + str(pars.depth) + '_batch_' + str(pars.train) + '_chain_' + str(pars.chains) + '_T_' + str(pars.T) + '_VR_' + str(pars.var_reduce) + '_p_' + str(pars.period) + '_burn_' + str(pars.burn) + '_seed_' + str(seed)
        #DIR = 'output/test_snapshot/' + pars.data + '_' + pars.model + str(pars.depth) + '_batch_' + str(pars.train) + '_chain_' + str(pars.chains) + '_T_' + str(pars.T) + '_VR_' + str(pars.var_reduce) + '_p_' + str(pars.period) + '_burn_' + str(pars.burn) + '_cycle_' + str(pars.cycle) + '_seed_' + str(seed)
    else:
        DIR = 'output/test_snapshot/' + pars.data + '_' + pars.model + str(pars.depth) + '_batch_' + str(pars.train) + '_chain_' + str(pars.chains) + '_T_' + str(pars.T) + '_VR_' + str(pars.var_reduce) + '_p_' + str(pars.period) + '_burn_' + str(pars.burn) + '_cycle_' + str(pars.cycle) + '_seed_' + str(seed)

    for filename in sorted(os.listdir(DIR)):
        if filename[-1] not in ['5']:
            continue
        file_idx = float(filename.split('_')[-1])
        cur_beta = (file_idx % sub_sn) * 1.0 / sub_sn
        if pars.cycle == 1 and cur_beta < 0.8:
                continue
        elif cur_beta < 0.7 or cur_beta in [0.94, 0.86, 0.78]:
                continue
        #elif cur_beta < 0.9:
        #        continue
        elif pars.cycle == 4 and ((file_idx <=500 and cur_beta < 0.8) or (file_idx >500 and cur_beta < 0.85)):
                continue
        net.load_state_dict(torch.load(DIR + '/' + filename))
        net.eval()

        if pars.chains == 2 and filename.startswith('Chain_0'):
            continue


        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs = net.forward(images).data / pars.T_scale
            prob = softmax(outputs)
            if idx == 1:
                output_ensemble_seen.append(outputs)
                prob_ensemble_seen.append(prob)
            else:
                output_ensemble_seen[cnt] = (1. - 1. / idx) * output_ensemble_seen[cnt] + (1. / idx) * outputs
                prob_ensemble_seen[cnt] = (1. - 1. / idx) * prob_ensemble_seen[cnt] + (1. / idx) * prob

        for cnt, (images, labels) in enumerate(target_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs = net.forward(images).data / pars.T_scale
            prob = softmax(outputs)
            if idx == 1:
                output_ensemble_unseen.append(outputs)
                prob_ensemble_unseen.append(prob)
            else:
                output_ensemble_unseen[cnt] = (1. - 1. / idx) * output_ensemble_unseen[cnt] + (1. / idx) * outputs
                prob_ensemble_unseen[cnt] = (1. - 1. / idx) * prob_ensemble_unseen[cnt] + (1. / idx) * prob
        
        idx += 1

        Brier_seen, counts_seen = 0, 0
        # entropy ranges from 0 to 2.5 roughly with each unit of width 0.05
        hist_brier_seen = [0] * 300000
        hist_entropy_seen = [0] * 50
        hist_entropy_unseen = [0] * 50
        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            prob_seen = prob_ensemble_seen[cnt]
            #prob_seen = softmax(output_ensemble_seen[cnt])
            one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()
            counts_seen += prob_seen.shape[0]
            Brier_seen += torch.mean((prob_seen - one_hot)**2,dim=1).sum().item()
            prob_seen_reg = prob_seen + 1e-20
            entropy_idx = (torch.sum(-prob_seen_reg * torch.log(prob_seen_reg), dim=1) / 0.05).int().tolist()
            for idx_ in entropy_idx:
                hist_entropy_seen[idx_] += 1
    
        Brier_unseen = 0
        counts_unseen = 0
        for cnt, (images, labels) in enumerate(target_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            prob_unseen = prob_ensemble_unseen[cnt] 
            #prob_unseen = softmax(output_ensemble_unseen[cnt])
            counts_unseen += prob_unseen.shape[0]
            Brier_unseen += torch.mean((prob_unseen)**2,dim=1).sum().item()
            prob_unseen_reg = prob_unseen + 1e-20
            entropy_idx = (torch.sum(-prob_unseen_reg * torch.log(prob_unseen_reg), dim=1) / 0.05).int().tolist()
            for idx_ in entropy_idx:
                hist_entropy_unseen[idx_] += 1
        print('===' * 100)
        print('Seed {} {} cur_beta {:.2f} Seen / Unseen / Total Brier score {:.4f} / {:.3f} / {:.3f}'.format(seed, filename, cur_beta, \
                Brier_seen/counts_seen, Brier_unseen/counts_unseen, (Brier_seen+Brier_unseen)/(counts_seen+counts_seen)))

        print("Entropy seen (from low to high)")
        print(hist_entropy_seen)
        print("Entropy unseen (from high to low)")
        print(hist_entropy_unseen[::-1])
