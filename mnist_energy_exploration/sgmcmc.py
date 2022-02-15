import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable

class Sampler:
    def __init__(self, net, pars):
        self.net = net
        self.lr = pars.lr
        self.T = pars.T
        self.wdecay = pars.wdecay
        self.N = pars.N
        """ Adaptive weighted """
        self.div = pars.div
        self.part = pars.part
        self.zeta = pars.zeta
        self.J = pars.part - 1
        self.gmul = 1.

        self.G = torch.cuda.FloatTensor(pars.part).fill_(1.) / pars.part
        self.import_weight = 1.
        print('Current Theta')
        print(self.G)
    
    def update_noise(self):
        return np.sqrt(2.0 * self.lr * self.T)

    def step(self, x, y):
        noise_scale = self.update_noise()
        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            param.data.add_(grads, alpha=-self.lr).add_(proposal)
    
    def cstep(self, x, y, decay, loss):
        """ Update energy PDFs """
        self.J = int(np.clip((loss) / self.div, 1, self.part-1))
        self.gmul = 1 + self.zeta * self.T * (torch.log(self.G[self.J]) - torch.log(self.G[self.J-1])) / self.div
        self.randomField = -self.G[self.J] * self.G
        self.randomField[self.J] = self.G[self.J] * (1. - self.G[self.J])
        self.G = self.G + decay * self.randomField

        noise_scale = self.update_noise()
        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            grads.mul_(self.gmul)
            param.data.add_(grads, alpha=-self.lr).add_(proposal)

        normalized_importance_weight = (self.G / self.G.max()) ** self.zeta
        self.import_weight = normalized_importance_weight[self.J].item()
        return self.import_weight

    def sgd(self, x, y):
        for i, param in enumerate(self.net.parameters()):
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            param.data.add_(grads, alpha=-self.lr)

