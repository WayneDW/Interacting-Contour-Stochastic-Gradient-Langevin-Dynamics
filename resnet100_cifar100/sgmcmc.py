"""
Code for Interacting-Contour-Stochastic-Gradient-Langevin-Dynamics
(c) Wei Deng @ Purdue University
Nov 8, 2021
"""

import sys
import numpy as np
import torch
import random
from torch.autograd import Variable

class Sampler:
    def __init__(self, net, pars):
        self.net = net
        self.lr = pars.lr
        self.momentum = pars.momentum
        self.T = pars.T
        self.wdecay = pars.wdecay
        self.V = 0.001
        self.velocity = []
        self.alpha = 1 - self.momentum
        self.N = pars.N
        """ Adaptive weighted """
        self.bias = pars.bias
        self.div = pars.div
        self.part = pars.part
        self.zeta = pars.zeta
        self.J = pars.part - 1
        self.gmul = 1.
        for param in net.parameters():
            p = torch.zeros_like(param.data)
            self.velocity.append(p)

        self.G = torch.cuda.FloatTensor(pars.part).fill_(1.) / pars.part
    
    def update_noise(self):
        beta = 0.5 * self.V * self.lr
        if beta > self.alpha:
            sys.exit('Momentum is too large')
        sigma = np.sqrt(2.0 * self.lr * (self.alpha - beta))
        noise_scale = sigma * np.sqrt(self.T)
        return noise_scale

    def update_H(self, randomField, stepsize):
        self.G = self.G + stepsize * randomField

    def step(self, x, y):
        noise_scale = self.update_noise()
        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            self.velocity[i].mul_(self.momentum).add_(grads, alpha=-self.lr).add_(proposal)
            param.data.add_(self.velocity[i])
    
    def cstep(self, x, y, loss, lowgrad, upgrad):
        """ Update energy PDFs/ density of states """
        gdrift = self.zeta * self.T * (torch.log(self.G[self.J]) - torch.log(self.G[self.J-1])) / self.div
        self.J = int(np.clip((loss - self.bias) / self.div, 1, self.part-1))
        """ regularization of gradient multiplier is adopted to avoid excessive moves """
        self.gmul = min(max(lowgrad, 1 + gdrift), upgrad) if self.J < self.part-1 else 1.

        self.randomField = -self.G[self.J] * self.G
        self.randomField[self.J] = self.G[self.J] * (1. - self.G[self.J])

        noise_scale = self.update_noise()
        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            grads.mul_(self.gmul)
            self.velocity[i].mul_(self.momentum).add_(grads, alpha=-self.lr).add_(proposal)
            param.data.add_(self.velocity[i])
        """ approximation of importance weights is adopted since zeta can be too large in DNN """
        return self.G[self.J].item() if self.J < self.part-1 else 0.

    def sgd(self, x, y):
        for i, param in enumerate(self.net.parameters()):
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            if self.momentum != 0:
                self.velocity[i].mul_(self.momentum).add_(grads)
                grads = self.velocity[i]
            param.data.add_(grads, alpha=-self.lr)
            

