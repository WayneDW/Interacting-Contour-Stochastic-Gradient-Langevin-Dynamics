#!/usr/bin/python
# from __future__ import print_function
import math
import copy
import sys
import os
import timeit
import csv
from tqdm import tqdm ## better progressbar
from math import exp

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as Func
import torch.nn as nn

from numpy import genfromtxt
from copy import deepcopy

"""
small models in our framework
"""
class CNN(nn.Module):
    def __init__(self, classes, filters, hidden): 
        super(CNN, self).__init__()       
        self.conv1 = nn.Conv2d(1, 12, 5, padding=2) 
        self.conv2 = nn.Conv2d(12, 12, 5, padding=2) 
        self.fc1 = nn.Linear(12*7*7, 10)
        self.fc2 = nn.Linear(10, classes)

    def convs(self, x):
        x = Func.max_pool2d(Func.relu(self.conv1.forward(x)), 2)
        x = Func.max_pool2d(Func.relu(self.conv2.forward(x)), 2) 
        return(x)

    def clf(self, x): 
        x = x.view(-1, 12*7*7)
        x = Func.relu(self.fc1.forward(x))  
        x = self.fc2.forward(x) 
        return(Func.log_softmax(x, dim=1))
    
    def forward(self, x):
        x = self.convs(x)
        return(self.clf(x))

class CNN5(nn.Module):
    def __init__(self, classes, filters, hidden):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, 5, padding=2)
        self.conv2 = nn.Conv2d(filters, filters*2, 5, padding=2)
        self.fc1 = nn.Linear(filters*2*7*7, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.filters = filters

    def convs(self, x):
        x = Func.max_pool2d(Func.relu(self.conv1.forward(x)), 2)
        x = Func.max_pool2d(Func.relu(self.conv2.forward(x)), 2)
        return(x)

    def clf(self, x):
        x = x.view(-1, self.filters*2*7*7)
        x = Func.relu(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return(Func.log_softmax(x, dim=1))

    def forward(self, x):
        x = self.convs(x)
        return(self.clf(x))
