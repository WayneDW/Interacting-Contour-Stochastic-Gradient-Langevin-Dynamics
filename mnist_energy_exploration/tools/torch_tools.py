import math
import numpy as np
import copy
import sys
import os
import timeit
import csv
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
from torchvision import datasets , transforms

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
         
import torch.utils.data as data
import torchvision.datasets as datasets
 
from copy import deepcopy
from sys import getsizeof

CUDA_EXISTS = torch.cuda.is_available()


def loader(train_size, test_size, args):
    if 1:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        exit('Unknown dataset')

    trainset = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=train_size, shuffle=True, num_workers=0) # num_workers=0 is crucial for seed
    """ caution: no shuffle on test dataset """
    testset = datasets.MNIST(root='./data/MNIST', train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=test_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def uncertainty_estimation(data, net, test_loader, prob_avg_seen, prob_avg_unseen, weight, acc_weights, counter, classes, print_tag=True, info='vanilla'):
    softmax = nn.Softmax(dim=1)
    for TT in prob_avg_seen:
        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            filters = (labels < classes)
            images, labels = images[filters], labels[filters]
            prob = softmax(net.forward(images).data / TT) * (weight + 1e-10)
            if counter == 1:
                prob_avg_seen[TT].append(prob)
            else:
                prob_avg_seen[TT][cnt] += prob

        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            filters = (labels >= classes)
            images, labels = images[filters], labels[filters]
            prob = softmax(net.forward(images).data / TT) * (weight + 1e-10)
            if counter == 1:
                prob_avg_unseen[TT].append(prob)
            else:
                prob_avg_unseen[TT][cnt] += prob

        Brier_seen, counts_seen = 0, 0
        hist_brier_seen = [0] * 300000
        """ MNIST-5 classes: entropy range 0 to 1.6 """
        hist_entropy_seen = [0] * 50
        hist_entropy_unseen = [0] * 50
        train_data_classes = 5
        width = 0.035
        calibration_true, calibration_false, calibration_counts, expected = [0] * 20, [0] * 20, [0] * 20, [0] * 20

        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            filters = (labels < classes)
            images, labels = images[filters], labels[filters]

            prob_seen = prob_avg_seen[TT][cnt] * 1. / acc_weights # normalized weighted prob

            for uidx in range(20):
                tag = ((prob_seen.max(1)[0] > width * uidx) & (prob_seen.max(1)[0] <= width * (uidx+1)))
                calibration_true[uidx] += (prob_seen.max(1)[1].eq(labels.data) & tag).sum().item()
                calibration_false[uidx] += ((1 - prob_seen.max(1)[1].eq(labels.data)) & tag).sum().item()
                calibration_counts[uidx] += tag.sum().item()
                expected[uidx] += prob_seen.max(1)[0][tag == 1].sum().item()

            one_hot = torch.nn.functional.one_hot(labels, num_classes=train_data_classes).float()
            counts_seen += prob_seen.shape[0]
            Brier_seen += torch.mean((prob_seen - one_hot)**2,dim=1).sum().item()
            prob_seen_reg = prob_seen + 1e-20
            entropy_idx = (torch.sum(-prob_seen_reg * torch.log(prob_seen_reg), dim=1) / width).int().tolist()
            for idx_ in entropy_idx:
                hist_entropy_seen[idx_] += 1

        Brier_unseen = 0
        counts_unseen = 0
        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            filters = (labels >= classes)
            images, labels = images[filters], labels[filters]

            prob_unseen = prob_avg_unseen[TT][cnt] * 1. / acc_weights
            counts_unseen += prob_unseen.shape[0]
            Brier_unseen += torch.mean((prob_unseen)**2,dim=1).sum().item()
            prob_unseen_reg = prob_unseen + 1e-20
            entropy_idx = (torch.sum(-prob_unseen_reg * torch.log(prob_unseen_reg), dim=1) / width).int().tolist()
            for idx_ in entropy_idx:
                hist_entropy_unseen[idx_] += 1

        def num_dig(x): return float(str(x)[:4])
        calibration_acc = [num_dig(calibration_true[i] * 100.0 / (calibration_true[i] + calibration_false[i] + 1e-10)) for i in range(20)]
        baseline_acc = [num_dig(expected[i] * 100.0 / (calibration_counts[i]+1e-10)) for i in range(20)]
        ECE = (np.abs(np.array(calibration_acc) - np.array(baseline_acc)) * np.array(calibration_counts) * 1.0 / sum(calibration_counts)).sum()
        if print_tag == True:
            print('\n' + '===' * 50)
            print('{} scaling {} Seen / Unseen / ECE {:.4f} / {:.5f} / {:.2f}'.format(info, TT, \
                    Brier_seen/counts_seen, Brier_unseen/counts_unseen, ECE))
            print("Entropy seen (from low to high)")
            print(hist_entropy_seen)
            print("Entropy unseen (from high to low)")
            print(hist_entropy_unseen[::-1])
            print('Calibration acc (top: base v.s. mid: proposed v.s. bottom: counts)')
            print(baseline_acc)
            print(calibration_acc)
            print(calibration_counts)


class BayesEval:
    def __init__(self, classes):
        data = 'mnist'
        self.counter = 1
        self.bma = []
        self.cur_acc = 0
        self.bma_acc = 0
        self.best_cur_acc = 0
        self.best_bma_acc = 0
        self.best_nll = float('inf')
        self.best_bma_nll = float('inf')
        self.acc_weights = 0
        self.classes = classes

        self.prob_avg_seen = {1: []}
        self.prob_avg_unseen = {1: []} 

    def eval(self, net, train_loader, test_loader, criterion, classes, weight=1,  bma=False):
        net.eval()
        """ evaluate exact loss """
        self.train_loss = 0
        for cnt, (images, labels) in enumerate(train_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            filters = (labels < self.classes)
            images, labels = images[filters], labels[filters]
            outputs = net.forward(images).data
            self.train_loss += (criterion(outputs, labels) * outputs.shape[0]).item()

        one_correct, bma_correct, self.test_loss, self.bma_nll = 0, 0, 0, 0
        """ Non-convex optimization """
        number_data_points = 0
        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            filters = (labels < self.classes)
            images, labels = images[filters], labels[filters]
            number_data_points += images.shape[0]
            outputs = net.forward(images).data
            self.test_loss += (criterion(outputs, labels) * outputs.shape[0]).item()
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
            uncertainty_estimation(data, net, test_loader, self.prob_avg_seen, self.prob_avg_unseen, weight, self.acc_weights, self.counter, classes)
            self.counter += 1
        self.cur_acc = 100.0 * one_correct / number_data_points
        self.bma_acc = 100.0 * bma_correct / number_data_points
        self.best_cur_acc = max(self.best_cur_acc, self.cur_acc)
        self.best_bma_acc = max(self.best_bma_acc, self.bma_acc)
        self.best_nll = min(self.best_nll, self.test_loss)
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
