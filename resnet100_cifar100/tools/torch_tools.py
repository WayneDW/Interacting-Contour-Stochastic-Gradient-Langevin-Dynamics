import math
import numpy as np
import copy
import sys
import os
import timeit
import csv
#import dill
from tqdm import tqdm ## better progressbar
from math import exp
import random
import pickle

import numpy as np
from numpy import genfromtxt


## import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn.functional as Func
import torch.nn as nn
from torchvision import datasets, transforms

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
         
import torch.utils.data as data
import torchvision.datasets as datasets
 
#import transforms 
from copy import deepcopy
from sys import getsizeof

CUDA_EXISTS = torch.cuda.is_available()

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

def loader(train_size, test_size, args):
    if args.data.startswith('cifar'):
        if args.data == 'cifar10':
            dataloader = datasets.CIFAR10
        else:
            dataloader = datasets.CIFAR100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        exit('Unknown dataset')

    trainset = dataloader('./data/' + args.data.upper(), train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=train_size, shuffle=True, num_workers=0) # num_workers=0 is crucial for seed
    """ caution: no shuffle on test dataset """
    testset = dataloader(root='./data/' + args.data.upper(), train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=test_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def uncertainty_estimation(data, net, test_loader, extra_loader, prob_avg_seen, prob_avg_unseen, weight, acc_weights, counter, print_tag=True, info='vanilla'):
    softmax = nn.Softmax(dim=1)
    for TT in prob_avg_seen:
        for cnt, (images, labels) in enumerate(extra_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            prob = softmax(net.forward(images).data / TT) * (weight + 1e-10)
            if counter == 1:
                prob_avg_unseen[TT].append(prob)
            else:
                prob_avg_unseen[TT][cnt] += prob

        Brier_unseen = 0
        counts_unseen = 0
        for cnt, (images, labels) in enumerate(extra_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            prob_unseen = prob_avg_unseen[TT][cnt] * 1. / acc_weights
            counts_unseen += prob_unseen.shape[0]
            Brier_unseen += torch.mean((prob_unseen)**2,dim=1).sum().item()

        if print_tag == True:
            print('\n' + '===' * 50)
            print('{} scaling {} Unseen  / {:.5f}'.format(info, TT, Brier_unseen/counts_unseen))


class BayesEval:
    def __init__(self, data):
        self.counter = 1
        self.bma = []
        self.cur_acc = 0
        self.bma_acc = 0
        self.best_cur_acc = 0
        self.best_bma_acc = 0
        self.best_nll = float('inf')
        self.best_bma_nll = float('inf')
        self.acc_weights = 0

        self.prob_avg_seen = {1: []} if data == 'cifar100' else {1: [], 2: []}
        self.prob_avg_unseen = {1: []} if data == 'cifar100' else {1: [], 2: []}

    def eval(self, data, net, test_loader, extra_loader, criterion, weight=1,  bma=False):
        net.eval()
        one_correct, bma_correct, self.nll, self.bma_nll = 0, 0, 0, 0
        """ Non-convex optimization """
        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs = net.forward(images).data
            self.nll += (criterion(outputs, labels) * outputs.shape[0]).item()
            one_correct += outputs.max(1)[1].eq(labels.data).sum().item()
            if bma == True:
                outputs = outputs * (weight + 1e-10)
                if cnt == 0:
                    self.acc_weights += (weight + 1e-10)
                if self.counter == 1:
                    self.bma.append(outputs)
                else:
                    self.bma[cnt] += outputs
                bma_correct += self.bma[cnt].max(1)[1].eq(labels.data).sum().item()
                self.bma_nll += (criterion(self.bma[cnt] * 1. / self.acc_weights, labels) * outputs.shape[0]).item()
        if bma == True:
            uncertainty_estimation(data, net, test_loader, extra_loader, self.prob_avg_seen, self.prob_avg_unseen, weight, self.acc_weights, self.counter)
            self.counter += 1
        self.cur_acc = 100.0 * one_correct / len(test_loader.dataset)
        self.bma_acc = 100.0 * bma_correct / len(test_loader.dataset)
        self.best_cur_acc = max(self.best_cur_acc, self.cur_acc)
        self.best_bma_acc = max(self.best_bma_acc, self.bma_acc)
        self.best_nll = min(self.best_nll, self.nll)
        self.best_bma_nll = min(self.best_bma_nll, self.bma_nll) if bma == True else float('inf')


class StochasticWeightAvg:
    def __init__(self, data, cycle=4):
        self.models = [None] * cycle
        self.models_square = [None] * cycle
        self.models_lowrank = [[] for _ in range(cycle)]

        """ swag part """
        self.best_cur_acc = 0
        self.best_bma_acc = 0
        self.best_cur_nll = float('inf')
        self.best_bma_nll = float('inf')
        self.bma = []
        self.prob_avg_seen = {1: []} if data == 'cifar100' else {1: [], 2: []}
        self.prob_avg_unseen = {1: []} if data == 'cifar100' else {1: [], 2: []}

        """ swag-diag """
        self.best_cur_acc_diag = 0
        self.best_bma_acc_diag = 0
        self.best_cur_nll_diag = float('inf')
        self.best_bma_nll_diag = float('inf')
        self.bma_diag = []
        self.prob_avg_seen_diag = {1: []} if data == 'cifar100' else {1: [], 2: []}
        self.prob_avg_unseen_diag = {1: []} if data == 'cifar100' else {1: [], 2: []}

        self.counter = [0] * cycle
        self.count_test = 1

    def update(self, cycle_idx, new_model, cur_beta=0., target_beta=0.9):
        if cur_beta < target_beta:
            return
        if self.models[cycle_idx] == None:
            self.models[cycle_idx] = pickle.loads(pickle.dumps(new_model))
            self.models_square[cycle_idx] = pickle.loads(pickle.dumps(new_model))
            new_pars = self.models[cycle_idx].parameters()
            for param in self.models_square[cycle_idx].parameters():
                param.data = next(new_pars).data ** 2
            self.models_lowrank[cycle_idx].append(pickle.loads(pickle.dumps(new_model)))
        else:
            new_pars = new_model.parameters()
            for param in self.models[cycle_idx].parameters():
                param.data = (param.data * self.counter[cycle_idx] + next(new_pars).data) / (self.counter[cycle_idx] + 1)

            new_pars = new_model.parameters()
            for param in self.models_square[cycle_idx].parameters():
                param.data = (param.data * self.counter[cycle_idx] + next(new_pars).data ** 2) / (self.counter[cycle_idx] + 1)

            self.models_lowrank[cycle_idx].append(pickle.loads(pickle.dumps(new_model)))
            mean_pars = self.models[cycle_idx].parameters()
            new_pars = new_model.parameters()
            for param in self.models_lowrank[cycle_idx][-1].parameters():
                param.data = (next(new_pars).data - next(mean_pars).data)

        self.counter[cycle_idx] += 1

    def random_models(self, model, model_square, models_lowrank):
        random_swag_diag = pickle.loads(pickle.dumps(model))
        mean_pars = model.parameters()
        square_pars = model_square.parameters()
        for param in random_swag_diag.parameters():
            """ clamp to avoid nan values """
            var = torch.clamp((next(square_pars).data - next(mean_pars).data ** 2), 0, float('inf'))
            param.data = param.data + var ** 0.5 * torch.cuda.FloatTensor(param.data.size()).normal_()


        random_swag = pickle.loads(pickle.dumps(model))
        mean_pars = model.parameters()
        square_pars = model_square.parameters()
        for param in random_swag.parameters():
            """ clamp to avoid nan values """
            var = torch.clamp((next(square_pars).data - next(mean_pars).data ** 2), 0, float('inf'))
            param.data = param.data + var ** 0.5 * torch.cuda.FloatTensor(param.data.size()).normal_() / np.sqrt(2.)

        
        #random_swag_lowrank = pickle.loads(pickle.dumps(model))
        for model_lowrank in models_lowrank:
            pars_lowrank = model_lowrank.parameters()
            for param in random_swag.parameters():
                param.data = param.data + next(pars_lowrank).data * np.random.normal() / np.sqrt(2 * (len(models_lowrank)-1)) 

        random_swag.eval()
        random_swag_diag.eval()
        return random_swag_diag, random_swag
    
    def inference(self, data, cycle_idx, train_loader, test_loader, extra_loader, criterion, lr, repeats=10):
        self.models[cycle_idx].eval()
        print('Generate random models in testing period for Bayesian model averaging')
        for counter in range(repeats):
            net_diag, net = self.random_models(self.models[cycle_idx], self.models_square[cycle_idx], self.models_lowrank[cycle_idx])

            """ eval SWAG-diag model """
            optimizer = torch.optim.SGD(net_diag.parameters(), lr=lr*50000, momentum=0.9, weight_decay=0.0005)
            net_diag.train()
            for _ in range(1):
                for idx, (images, labels) in enumerate(train_loader):
                    images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
                    labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
                    loss = criterion(net_diag(images), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            net_diag.eval()
            one_correct_diag, bma_correct_diag, bma_nll_diag = 0, 0, 0
            for cnt, (images, labels) in enumerate(test_loader):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                outputs = net_diag.forward(images).data
                one_correct_diag += outputs.max(1)[1].eq(labels.data).sum().item()
                if self.count_test == 1:
                    self.bma_diag.append(outputs)
                else:
                    self.bma_diag[cnt] += outputs
                bma_correct_diag += self.bma_diag[cnt].max(1)[1].eq(labels.data).sum().item()
                bma_nll_diag += (criterion(self.bma_diag[cnt] * 1. / self.count_test, labels) * outputs.shape[0]).item()
            cur_one_acc_diag = 100.0 * one_correct_diag / len(test_loader.dataset)
            cur_bma_acc_diag = 100.0 * bma_correct_diag / len(test_loader.dataset)
            self.best_cur_acc_diag = max(self.best_cur_acc_diag, cur_one_acc_diag)
            self.best_bma_acc_diag = max(self.best_bma_acc_diag, cur_bma_acc_diag)
            self.best_bma_nll_diag = min(self.best_bma_nll_diag, bma_nll_diag)
            print('SWAG-diag model {} cur acc {:0.2f} BMA acc {:0.2f} Best acc {:0.2f} Best BMA: {:0.2f} NLL {:.1f} Best BMA NLL: {:.1f}'.format(\
                    counter, cur_one_acc_diag, cur_bma_acc_diag, self.best_cur_acc_diag, self.best_bma_acc_diag, bma_nll_diag, self.best_bma_nll_diag))
            if counter < repeats - 1:
                uncertainty_estimation(data, net_diag, test_loader, extra_loader, self.prob_avg_seen_diag, self.prob_avg_unseen_diag, 1., self.count_test, self.count_test, print_tag=False, info='swag-diag')
            else:
                uncertainty_estimation(data, net_diag, test_loader, extra_loader, self.prob_avg_seen_diag, self.prob_avg_unseen_diag, 1., self.count_test, self.count_test, print_tag=True, info='swag-diag')


            """ eval SWAG model """
            optimizer = torch.optim.SGD(net.parameters(), lr=lr*50000, momentum=0.9, weight_decay=0.0005)
            net.train()
            for _ in range(1):
                for idx, (images, labels) in enumerate(train_loader):
                    images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
                    labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
                    loss = criterion(net(images), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            net.eval()
            one_correct, bma_correct, bma_nll = 0, 0, 0
            for cnt, (images, labels) in enumerate(test_loader):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                outputs = net.forward(images).data
                one_correct += outputs.max(1)[1].eq(labels.data).sum().item()
                if self.count_test == 1:
                    self.bma.append(outputs)
                else:
                    self.bma[cnt] += outputs
                bma_correct += self.bma[cnt].max(1)[1].eq(labels.data).sum().item()
                bma_nll += (criterion(self.bma[cnt] * 1. / self.count_test, labels) * outputs.shape[0]).item()
            cur_one_acc = 100.0 * one_correct / len(test_loader.dataset)
            cur_bma_acc = 100.0 * bma_correct / len(test_loader.dataset)
            self.best_cur_acc = max(self.best_cur_acc, cur_one_acc)
            self.best_bma_acc = max(self.best_bma_acc, cur_bma_acc)
            self.best_bma_nll = min(self.best_bma_nll, bma_nll)
            print('SWAG      model {} cur acc {:0.2f} BMA acc {:0.2f} Best acc {:0.2f} Best BMA: {:0.2f} NLL {:.1f} Best BMA NLL: {:.1f}\n'.format(\
                    counter, cur_one_acc, cur_bma_acc, self.best_cur_acc, self.best_bma_acc, bma_nll, self.best_bma_nll))
            if counter < repeats - 1:
                uncertainty_estimation(data, net, test_loader, extra_loader, self.prob_avg_seen, self.prob_avg_unseen, 1., self.count_test, self.count_test, print_tag=False, info='swag')
            else:
                uncertainty_estimation(data, net, test_loader, extra_loader, self.prob_avg_seen, self.prob_avg_unseen, 1., self.count_test, self.count_test, print_tag=True, info='swag')

            self.count_test += 1
