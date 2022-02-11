import sys
import argparse
import random
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import copy 
from models import bb_net_rl, dropout_rl
import trainer
import processing
from samplers import Sampler
import torch


parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-sn', default=1000, type=int, help='number of epochs')
parser.add_argument('-c', default='sgld', type=str, help='type of algorithms')
parser.add_argument('-lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('-T', default=0, type=float, help='temperature')
parser.add_argument('-sz', default=0, type=float, help='stepsize for stochastic approximation')
parser.add_argument('-zeta', default=0, type=float, help='zeta')
parser.add_argument('-reg', default=1e-3, type=float, help='regularizer')
parser.add_argument('-wdecay', default=1e-2, type=float, help='L2 penalty')
parser.add_argument('-part', default=200, type=int, help='The number of partitions')
parser.add_argument('-div', default=2, type=float, help='Divide energy: divisor to calculate partition index')

parser.add_argument('-chains', default=1, type=int, help='Total number of chains')
parser.add_argument('-repeat', default=5, type=int, help='Total number of repeats')
parser.add_argument('-hidden', default=100, type=int, help='Number of hidden nodes')

parser.add_argument('-init', default=1024, type=int, help='burn in number of mushrooms')
parser.add_argument('-buffers', default=4096, type=int, help='buffer size')
parser.add_argument('-batch', default=512, type=int, help='Training batch size')
parser.add_argument('-trains', default=16, type=int, help='Train iterations')
parser.add_argument('-pull', default=20, type=int, help='Number of pulls')

""" hyperparameters for preconditioned SGLD """
parser.add_argument('-precondition', default=0, type=int, help='set preconditioner or not')
parser.add_argument('-preg', default=1e-3, type=float, help='regularizer for preconditioner')
parser.add_argument('-alpha', default=0.01, type=float, help='stepsize for preconditioner')

""" hyperparameters for dropout """
parser.add_argument('-rate', default=0, type=float, help='dropout rate')
parser.add_argument('-samples', default=1, type=int, help='repeating samples')

""" hyperparameters for RMSProp """
parser.add_argument('-decay', default=1.0, type=float, help='LR decay')
parser.add_argument('-epsilon', default=0, type=float, help='epsilon greedy')

""" hyperparameters for BayesBackProp """
parser.add_argument('-sigma1', default=1, type=float, help='')
parser.add_argument('-sigma2', default=1e-6, type=float, help='')
parser.add_argument('-sigma', default=0.02, type=float, help='')
parser.add_argument('-pi', default=0.5, type=float, help='')



parser.add_argument('-warm', default=0.1, type=float, help='warm up for CSGLD')
parser.add_argument('-seed', default=random.randint(1, 1e4), type=int, help='Random Seed')
parser.add_argument('-gpu', default=-1, type=int, help='Default GPU')
pars = parser.parse_args()
print(pars)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

""" set random seeds """
torch.manual_seed(pars.seed)
torch.cuda.manual_seed(pars.seed)
np.random.seed(pars.seed)
random.seed(pars.seed)
torch.backends.cudnn.deterministic=True

try:
    torch.cuda.set_device(pars.gpu)
except: # in case the device has only one GPU
    torch.cuda.set_device(0)


CUDA_EXISTS = torch.cuda.is_available() and pars.gpu >= 0
print('Using GPU: {}'.format(CUDA_EXISTS))
train_X, train_y = processing.get_mushroom()
dim_input = train_X.shape[1]
dim_action_space = 2


X_train_tensor = torch.from_numpy(train_X.copy()).float().unsqueeze(dim=1)
y_train_tensor = torch.from_numpy(train_y.copy()).float()

if CUDA_EXISTS:
    X_train_tensor, y_train_tensor = X_train_tensor.cuda(), y_train_tensor.cuda()

epsilons = [pars.epsilon]

regrets = np.zeros((len(epsilons), pars.sn + 1))
regrets_std = np.zeros_like(regrets)
for i, epsilon in enumerate(epsilons):
    regs = np.zeros((pars.repeat, pars.sn + 1))
    for run in range(pars.repeat):
        print('epsilon greedy {:.3f} sample {} / {}'.format(epsilon, run + 1, pars.repeat))
        nets, samplers = [], []
        for _ in range(pars.chains):
            if pars.c in ('sgd', 'sgld', 'csgld'):
                net = trainer.DeterministicRLNet(dim_input, pars.hidden, dim_action_space)
            elif pars.c == 'dropout':
                net = trainer.DropoutNet(dim_input, pars.hidden, dim_action_space, p=pars.rate)
            elif pars.c == 'bayesbackprop':
                prior_parameters = {'sigma1': pars.sigma1, 'sigma2': pars.sigma2, 'pi': pars.pi}
                net = trainer.BayesBackpropRLNet(dim_input, pars.hidden, dim_action_space, prior_parameters, sigma=pars.sigma)
            if CUDA_EXISTS:
                net = net.cuda()
            nets.append(net)

        """ ensemble net for predictions of actions """
        if pars.c in ('sgd', 'sgld', 'csgld'):
            agents = trainer.AgentGreedy(nets, epsilon)
            rl_reg = trainer.DeterministicRLReg(X_train_tensor, y_train_tensor, agents, \
                        buffer_size=pars.buffers, minibatch_size=pars.batch, burn_in=pars.init)
        elif pars.c == 'dropout':
            agents = trainer.AgentDropout(nets, sample=pars.samples)
            rl_reg = trainer.DropoutRLReg(X_train_tensor, y_train_tensor, agents, \
                        buffer_size=pars.buffers, minibatch_size=pars.batch, burn_in=pars.init)
        elif pars.c == 'bayesbackprop':
            agents = trainer.AgentBayesBackprop(nets, sample=pars.samples)
            rl_reg = trainer.BayesRLReg(X_train_tensor, y_train_tensor, agents, \
                        buffer_size=pars.buffers, minibatch_size=pars.batch, burn_in=pars.init)

        for idx in range(pars.chains):
            if pars.c == 'sgd':
                sampler = torch.optim.SGD(nets[idx].parameters(), lr=pars.lr, weight_decay=pars.wdecay)
            elif pars.c in ('sgld', 'csgld', 'dropout', 'bayesbackprop'):
                sampler = Sampler(nets[idx], pars, CUDA_EXISTS)
            else:
                sys.exit('Unknown algorithms.')
            samplers.append(sampler)
        rl_reg.train(pars.sn, pars.pull, pars.trains, pars.buffers, pars.sz, pars.warm, pars.decay, samplers, CUDA_EXISTS)
        regs[run] = copy.copy(rl_reg.hist['regret'])
        if pars.c == 'csgld':
            print('print G function (related to PDF in energy or density of states)')
            print(samplers[0].G.numpy())
    regrets[i] = regs.mean(axis=0)
    regrets_std[i] = regs.std(axis=0)


'''
plt.figure(figsize=(6, 4))
for epsilon, regs, regs_std in zip(epsilons, regrets, regrets_std):
    plt.plot(regs, label=r"$\epsilon$ = {}".format(epsilon))
    plt.fill_between(np.arange(pars.sn + 1), regs - regs_std, regs + regs_std, alpha=0.3)

plt.legend()
plt.xlabel('Step')
plt.ylabel('Cumulative regret')
plt.show()

plt.savefig('./figures/mushroom_' + pars.c + '_chains_' + str(pars.chains) + '_node_' + str(pars.hidden) + '_lr_' + str(pars.lr) + '_sz_' + str(pars.sz) \
        + '_T_' + str(pars.T) + '_l2_' + str(pars.wdecay)  + '_zeta_' + str(pars.zeta) + '_sn_' + str(pars.sn) + '_pull_' + str(pars.pull) \
        + '_trains_' + str(pars.trains) + '_div_' + str(pars.div) + '_part_' + str(pars.part) + '_repeat_' + str(pars.repeat) + '_seed_' + str(pars.seed) +  '.png', dpi=200)
'''
