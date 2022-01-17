#!/usr/bin/python3
import torchvision.datasets as datasets
import torch
import datetime
import time
from DE.JADE import JADE_MLP
from torch.utils.tensorboard import SummaryWriter
import threading

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

trainingt = torch.nn.functional.one_hot(mnist_trainset.targets,num_classes=10)
validationt = torch.nn.functional.one_hot(mnist_testset.targets,num_classes=10)


for maxdepth in range(4,16,2):
    for maxneuron in range(4,16,2):
        jade_tb = SummaryWriter('./DataCollector/DE_JADE/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_maxdepth_'+str(maxdepth)+'_maxneuron_'+str(maxneuron))             
        jade = JADE_MLP(outdim=10,maxdepth=maxdepth,mindepth=2,minneuron=2,maxneuron=maxneuron,initSize=10,trainingset=mnist_trainset.data, \
                         validationset=mnist_testset.data, trainingTarget=trainingt,validateTarget=validationt,crossover=2,tb=jade_tb)
        start=time.time()
        print(f'JADE Test Run Start: maxdepth:{maxdepth}, maxneuron:{maxneuron}')
        jade.run()
        end=time.time()
        print(f'JADE Test Run End: maxdepth:{maxdepth}, maxneuron:{maxneuron} runtime:{(end-start):10.8f}')
        