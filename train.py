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
torch.manual_seed(5)
from model_defs import *
def simple():
    model = nn.Sequential(
        nn.Conv2d(1, 1, 2, stride=1, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(4,2)
    )
    return model

if __name__ == '__main__':
    model=simple()
    #model.load_state_dict(torch.load('./mnist_crown/cnn_2layer_width_1_best.pth'))
    print(model)
    model = BoundSequential.convert(model)
    c=torch.eye(2)
    print('CROWN BOUND',model.full_backward_range(norm=np.inf, x_U=torch.zeros((1,1,3,3))-3, x_L=torch.zeros((1,1,3,3))+5, eps=0.0001, C=c, upper=True, lower=True))

