import sys
import numpy as np
import torch
import random
from torch.autograd import Variable

class Sampler:
    def __init__(self, net, pars, CUDA_EXISTS):
        self.net = net
        self.c = pars.c
        self.lr = pars.lr
        self.T = pars.T
        self.wdecay = pars.wdecay
        self.N = pars.buffers
        """ Adaptive weighted """
        self.div = pars.div
        self.part = pars.part
        self.zeta = pars.zeta
        self.reg = pars.reg
        self.J = pars.part - 1
        self.gmul = 1.
        self.gmul_min = 1.
        self.gmul_max = 1.
        self.CUDA_EXISTS = CUDA_EXISTS

        if self.CUDA_EXISTS:
            self.G = torch.cuda.FloatTensor(pars.part).fill_(1.) / pars.part
        else:
            self.G = torch.FloatTensor(pars.part).fill_(1.) / pars.part
        self.randomField = -self.G[self.J] * self.G

        """ default setup for preconditioner """
        self.precondition = pars.precondition
        self.batch = pars.batch
        self.alpha = pars.alpha
        self.preg = pars.preg
        self.Fisher_info = []
        for param in net.parameters():
            p = torch.ones_like(param.data)
            self.Fisher_info.append(p)
    
    def update_noise(self):
        return np.sqrt(2.0 * self.lr * self.T)

    def update_H(self, randomField, stepsize):
        self.G = self.G + stepsize * randomField

    def zero_grad(self):
        self.net.zero_grad()

    def step(self, loss=0):
        if self.c == 'csgld':
            """ Update energy PDFs """
            gdrift = self.zeta * self.T * (torch.log(self.G[self.J]) - torch.log(self.G[self.J-1])).item() / self.div
            self.gmul = 1 + gdrift if self.J < self.part-1 else 1.
            self.gmul_min = min(self.gmul_min, self.gmul)
            self.gmul_max = max(self.gmul_max, self.gmul)
            if self.J < self.part-1:
                self.randomField = -self.G[self.J] * self.G
                self.randomField[self.J] = self.G[self.J] * (1. - self.G[self.J])

            self.J = int(np.clip(loss / self.div, 1, self.part-1))

        noise_scale = self.update_noise()
        for layer, param in enumerate(self.net.parameters()):
            if self.CUDA_EXISTS:
                proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            else:
                proposal = torch.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            grads.mul_(self.gmul)
            if self.precondition:
                self.Fisher_info[layer].mul_(1 - self.alpha).add_(self.alpha * ((grads/self.batch)**2))
                preconditioner = torch.div(1., self.preg + torch.sqrt(self.Fisher_info[layer]))
                grads.mul_(preconditioner)
                proposal.mul_(torch.sqrt(preconditioner))
            param.data.add_(grads, alpha=-self.lr).add_(proposal)
        return self.G[self.J].item() ** self.zeta + self.reg if self.J < self.part-1 else self.reg
