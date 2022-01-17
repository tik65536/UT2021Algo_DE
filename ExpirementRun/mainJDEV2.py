#!/usr/bin/python3
import torchvision.datasets as datasets
import torch
import datetime
import time
from DE.JDE import JDEV2_MLP
from torch.utils.tensorboard import SummaryWriter

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

trainingt = torch.nn.functional.one_hot(mnist_trainset.targets,num_classes=10)
validationt = torch.nn.functional.one_hot(mnist_testset.targets,num_classes=10)


for maxdepth in range(4,16,2):
    for maxneuron in range(4,16,2):
        jdev2_tb = SummaryWriter('./DataCollector/DE_JDEV2/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_maxdepth_'+str(maxdepth)+'_maxneuron_'+str(maxneuron))
        jdev2 = JDEV2_MLP(outdim=10,maxdepth=maxdepth,mindepth=2,minneuron=2,maxneuron=maxneuron,initSize=10,trainingset=mnist_trainset.data, \
                         validationset=mnist_testset.data, trainingTarget=trainingt,validateTarget=validationt,crossover=2,tb=jdev2_tb)
        start=time.time()
        print(f'JDEV2 Test Run Start: maxdepth:{maxdepth}, maxneuron:{maxneuron}')
        jdev2.run()
        end=time.time()
        print(f'JDEV2 Test Run End: maxdepth:{maxdepth}, maxneuron:{maxneuron} runtime:{(end-start):10.8f}')
        