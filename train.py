## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import sys
import copy
import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
import numpy as np
from bound_layers import BoundSequential, BoundLinear, BoundConv2d, BoundDataParallel
import torch.optim as optim
# from gpu_profile import gpu_profile
import time
from datetime import datetime
#previous case seed
#torch.manual_seed(5)

#max pooling case seed
torch.manual_seed(100)
from model_defs import *


"""
def simple():
    model = nn.Sequential(
        nn.Conv2d(1, 1, 2, stride=1, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(4,2)
    )
    return model
"""


def simple():
    model = nn.Sequential(
        nn.Conv2d(1, 1, 2, stride=1, padding=0),
        nn.ReLU(),
        #TODO: Check why stride equals 2 in IBP
        nn.MaxPool2d(kernel_size = 2, stride=1),
        Flatten(),
        nn.Linear(1,1)
    )
    return model

def complex():
    model = nn.Sequential(
        nn.Conv2d(3, 3, 2, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(3, 3, 2, stride=1, padding=0),
        nn.ReLU(),
        #TODO: Check why stride equals 2 in IBP
        nn.MaxPool2d(kernel_size = 2, stride=1),
        Flatten(),
        nn.Linear(1,1)
    )
    return model
#input model, x_U, x_L, output_dimension, norm=np.inf
#return CROWN upper bound and lower bound
#See main as example
def CROWN_with_max_pooling(model, x_U, x_L, output_dimension, norm=np.inf):
    c = torch.eye(output_dimension)
    model = BoundSequential.convert(model)
    crown_output=model.full_backward_range(norm=norm, x_U=x_U, x_L=x_L, eps=0.0001, C=c, upper=True, lower=True)
    return crown_output[0],crown_output[2]



if __name__ == '__main__':
    model=simple()
    #model.load_state_dict(torch.load('./mnist_crown/cnn_2layer_width_1_best.pth'))
    x_U = torch.rand((1, 1, 3, 3))
    x_L = torch.rand((1, 1, 3, 3)) - 1
    print('xU, xL here', x_U, x_L, x_U-x_L)
    print(model)
    print('model output', model(torch.zeros((1, 1, 3, 3)) - 3))

    model = BoundSequential.convert(model)
    c=torch.eye(1)

    print('CROWN BOUND',model.full_backward_range(norm=np.inf, x_U=x_U, x_L=x_L, eps=0.0001, C=c, upper=True, lower=True))
    print('CROWN BOUND return', 'ub, upper_sum_b, lb, lower_sum_b')

