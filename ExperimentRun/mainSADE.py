#!/usr/bin/python3
import torchvision.datasets as datasets
import torch
import datetime
import time
from DE.SADE import SADE_MLP
from torch.utils.tensorboard import SummaryWriter

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

trainingt = torch.nn.functional.one_hot(mnist_trainset.targets,num_classes=10)
validationt = torch.nn.functional.one_hot(mnist_testset.targets,num_classes=10)


for maxdepth in range(4,16,2):
    for maxneuron in range(4,16,2):
        sade_tb = SummaryWriter('./DataCollector/DE_SADE/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_maxdepth_'+str(maxdepth)+'_maxneuron_'+str(maxneuron))            
        sade = SADE_MLP(outdim=10,maxdepth=maxdepth,mindepth=2,minneuron=2,maxneuron=maxneuron,initSize=10,trainingset=mnist_trainset.data, \
                         validationset=mnist_testset.data, trainingTarget=trainingt,validateTarget=validationt,crossover=2,tb=sade_tb)
        start=time.time()
        print(f'SADE Test Run Start: maxdepth:{maxdepth}, maxneuron:{maxneuron}')
        sade.run()
        end=time.time()
        print(f'SADE Test Run End: maxdepth:{maxdepth}, maxneuron:{maxneuron} runtime:{(end-start):10.8f}')
        