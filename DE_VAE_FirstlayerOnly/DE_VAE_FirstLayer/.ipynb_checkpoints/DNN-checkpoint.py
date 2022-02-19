#!/usr/bin/python3
import torch
from collections import OrderedDict
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        # depth of the neural network and activation function
        self.depth = len(layers)-1
        self.activation = torch.nn.ReLU
        # layers: (depth-1)x 'linear' + 'activation' + last one 'linear'
        layer_list = list()
        for i in range(self.depth-1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
