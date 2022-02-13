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
from tools import BayesEval, StochasticWeightAvg
from sgmcmc import Sampler

CUDA_EXISTS = torch.cuda.is_available()

def sgmcmc(nets, train_loader, test_loader, extra_loader, pars):
    start = timeit.default_timer()
    samplers, BMA, SWAG = [], BayesEval(pars.data), StochasticWeightAvg(pars.data, cycle=pars.cycle)
    criterion = nn.CrossEntropyLoss()
    for idx in range(pars.chains):
        samplers.append(Sampler(nets[idx], pars))

    import_w =  [0.] * pars.chains
    """ Initialization for variance reduction """
    last_full_losses, last_VRnets = [0] * pars.chains, [pickle.loads(pickle.dumps(nets[i])) for i in range(pars.chains)]

    """ Initialization for cyclic and swag """
    cur_beta, sub_sn = 0, pars.sn // pars.cycle
    BMAS = [BayesEval(pars.data) for _ in range(pars.chains)]

    """ Initialization for replica exchange """
    counter, cooling, cumulative_swap = 0, [[]] * (pars.chains-1), [0.] * (pars.chains)
    
    if pars.c not in ['sghmc', 'csghmc', 'cswag', 'replica']:
        exit('Unknown training method')
    for epoch in range(pars.sn):
        """ update adaptive variance and variance reduction every [period] epochs """
        if pars.period > 0 and epoch % pars.period == 0 and epoch >= pars.warm * pars.sn - pars.period:
            last_full_losses = [0] * pars.chains
            for idx in range(pars.chains):
                nets[idx].eval()
                for cnt_batches, (images, labels) in enumerate(train_loader):
                    images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
                    labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
                    last_full_losses[idx] += (criterion(nets[idx](images), labels).item() * pars.batch)
                last_VRnets[idx] = pickle.loads(pickle.dumps(nets[idx]))

        if pars.cycle >= 2 or pars.c == 'sghmc' or (pars.c == 'csghmc' and pars.model == 'wrn'):
            cur_beta = (epoch % sub_sn) * 1.0 / sub_sn
            for idx in range(pars.chains):
                """ constant learning rate during exploration """
                samplers[idx].lr = pars.lr / 2 * (np.cos(np.pi * min(cur_beta, pars.warm)) + 1)
                if (epoch % sub_sn) * 1.0 / sub_sn == 0:
                    print('Chain {} Cooling down for optimization'.format(idx))
                    samplers[idx].T /= 1e10
                elif epoch % sub_sn == int(pars.warm * sub_sn):
                    print('Chain {} Heating up for sampling'.format(idx))
                    samplers[idx].T *= 1e10
        elif epoch in [int(0.5 * pars.sn), int(0.75 * pars.sn)]:
            for idx in range(pars.chains):
                samplers[idx].lr *= pars.LRanneal
            
            if pars.c == 'replica' and epoch == int(0.5 * pars.sn):
                for idx in range(pars.chains):
                    samplers[idx].T *= (pars.scalar_T ** (pars.chains-1-idx))

        if pars.c == 'replica' and epoch == 0:
            for idx in range(pars.chains):
                samplers[idx].lr *= (pars.scalar_lr ** (pars.chains-1-idx))

        each_loss = [0] * pars.chains
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
            for idx in range(pars.chains):
                nets[idx].train()
                samplers[idx].net.zero_grad()
                loss = criterion(samplers[idx].net(images), labels) * pars.N
                loss.backward()

                each_loss[idx] = loss.item()
                if pars.period > 0 and epoch > pars.warm * pars.sn:
                    control_variate_loss = criterion(last_VRnets[idx](images), labels).item() * pars.N
                    each_loss[idx] = each_loss[idx] - control_variate_loss + last_full_losses[idx]
                if pars.c != 'csghmc' or epoch <= pars.warm * pars.sn:
                    samplers[idx].step(images, labels)
                else:
                    import_w[idx] = samplers[idx].cstep(images, labels, each_loss[idx], pars.lowgrad, pars.upgrad)

            if pars.c == 'csghmc' and epoch > pars.warm * pars.sn:
                randomField = samplers[0].randomField
                for idx in range(1, pars.chains):
                    randomField = randomField * idx/ (idx+1) + samplers[idx].randomField / (idx+1)
                for idx in range(pars.chains):
                    samplers[idx].update_H(randomField, pars.stepsize)
            elif pars.c == 'replica' and epoch > pars.warm * pars.sn:
                counter += 1
                for idx in range(pars.chains - 1):
                    if each_loss[idx] + pars.correction < each_loss[idx+1] and epoch not in cooling[idx]:
                        temporary = pickle.loads(pickle.dumps(nets[idx+1]))
                        nets[idx+1].load_state_dict(nets[idx].state_dict())
                        nets[idx].load_state_dict(temporary.state_dict())
                        each_loss[idx], each_loss[idx+1] = each_loss[idx+1], each_loss[idx]
                        cumulative_swap[idx] += 1.
                        cooling[idx] = range(epoch, epoch+1)
                        print('Epoch {} Swap chain {} with chain {} Cumulative swaps {} counter {} swap rate {:.1e} Correction {:.1e}'.format(\
                                epoch, idx, idx+1, int(cumulative_swap[idx]), int(counter), cumulative_swap[idx] / counter, pars.correction))

        """ apply weight 1 using SGHMC and importance weights using Contour SGHMC """
        for idx in range(pars.chains):
            if pars.c == 'cswag':
                SWAG.update(epoch // sub_sn, nets[idx], cur_beta, pars.burn)
                if (epoch + 1) % sub_sn == 0 and epoch >= sub_sn - 1:
                    SWAG.inference(pars.data, epoch // sub_sn, train_loader, test_loader, extra_loader, criterion, samplers[idx].lr, repeats=10)

            if (pars.c in ['sghmc', 'replica', 'csghmc'] and epoch >= pars.burn * pars.sn) or \
                    (pars.c == 'cswag' and epoch % sub_sn >= pars.burn * sub_sn):
                if pars.c == 'replica':
                    BMAS[idx].eval(pars.data, nets[idx], test_loader, extra_loader, criterion, weight=import_w[idx], bma=True)
                else:
                    BMA.eval(pars.data, nets[idx], test_loader, extra_loader, criterion, weight=import_w[idx], bma=True)
            else:
                if pars.c == 'replica':
                    BMAS[idx].eval(pars.data, nets[idx], test_loader, extra_loader, criterion, weight=import_w[idx], bma=False)
                else:
                    BMA.eval(pars.data, nets[idx], test_loader, extra_loader, criterion, weight=import_w[idx], bma=False)


            Gcum = samplers[idx].G.cpu().numpy()
            if pars.c.startswith('csghmc') and (epoch + 1) % 50 == 0:
                print('----------- Adaptive weights -----------')
                print('theta')
                print(np.array(Gcum))
                print('Grad mul')
                print(1 + samplers[idx].zeta * samplers[idx].T * (np.log(Gcum[1:]) - np.log(Gcum[:-1])) / samplers[idx].div)
            if pars.c == 'replica':
                print('Epoch {} Chain {} Acc: {:0.2f} BMA: {:0.2f} Best Acc: {:0.2f} Best BMA: {:0.2f} lr: {:.2E} T: {:.2E}  Loss: {:0.1f}'.format(\
                    epoch, idx, BMAS[idx].cur_acc,  BMAS[idx].bma_acc, BMAS[idx].best_cur_acc, BMAS[idx].best_bma_acc, samplers[idx].lr,  samplers[idx].T, each_loss[idx]))
                print('Epoch {} Chain {} NLL: {:.1f} Best NLL: {:.1f} BMA NLL: {:.1f}  Best BMA NLL: {:.1f}'.format(epoch, idx, BMAS[idx].nll, BMAS[idx].best_nll, BMAS[idx].bma_nll, BMAS[idx].best_bma_nll))
            else:
                print('Epoch {} Chain {} Acc: {:0.2f} BMA: {:0.2f} Best Acc: {:0.2f} Best BMA: {:0.2f} lr: {:.2E} T: {:.2E}  Weight {:.3f} Grad mul {:.2f} Pidx {} Loss: {:0.1f}'.format(\
                    epoch, idx, BMA.cur_acc,  BMA.bma_acc, BMA.best_cur_acc, BMA.best_bma_acc, samplers[idx].lr,  samplers[idx].T, import_w[idx], samplers[idx].gmul, samplers[idx].J, each_loss[idx]))
                print('Epoch {} Chain {} NLL: {:.1f} Best NLL: {:.1f} BMA NLL: {:.1f}  Best BMA NLL: {:.1f}'.format(epoch, idx, BMA.nll, BMA.best_nll, BMA.bma_nll, BMA.best_bma_nll))
        print('')
        
    end = timeit.default_timer()
    print("Sampling Time used: {:0.1f}".format(end - start))
