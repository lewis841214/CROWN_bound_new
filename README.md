# CROWN with MaxPooling

## Intro

This code is pytorch code based on CROWN part in CROWN-IBP provided by [**huanzhang12**](https://github.com/huanzhang12/CROWN-IBP/commits?author=huanzhang12)(Author of CROWN-IBP). But the code provided only allowed using Conv2d and Linear, ReLU, Flatten layers. I add the MaxPooling layer into their code, which allow one to use MaxPool2d layer in the sequential model.



## **Getting Started with the Code**

This program is tested on Pytorch 1.6.0 and Python 3.7.

To compute the CROWN bound of your model, first write your model in a Sequential manner, for example:

    model = nn.Sequential(
        nn.Conv2d(1, 1, 2, stride=1, padding=0),
        nn.ReLU(),
        #TODO: Check why stride equals 2 in IBP
        nn.MaxPool2d(kernel_size = 2, stride=1),
        Flatten(),
        nn.Linear(1,1)
    )

Then set your input permutation range, for axample:

    x_U = torch.rand((1, 1, 3, 3))
    x_L = torch.rand((1, 1, 3, 3)) - 1

Then set the output dimension of your model:


    output_dimension=1

Then import the function (CROWN_with_max_pooling) form train.py:

    from train.py import *
    CROWN_with_max_pooling(model, x_U, x_L, output_dimension, norm=np.inf)

Then the function will return the upper bound and lower bound of CROWN.






