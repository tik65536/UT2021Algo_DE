#!/usr/bin/python3
import torchvision.datasets as datasets
import torch
from DE import DE_MLP

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

trainingt = torch.nn.functional.one_hot(mnist_trainset.targets,num_classes=10)
validationt = torch.nn.functional.one_hot(mnist_testset.targets,num_classes=10)

d = DE_MLP(outdim=10,maxdepth=10,initSize=3,trainingset=mnist_trainset.data, validationset=mnist_testset.data, trainingTarget=trainingt,validateTarget=validationt)
#d.runMP(nump=4)
d.run()
