#!/usr/bin/python3
import torchvision.datasets as datasets
import torch
import datetime
import time
import pickle
import numpy as np
from DE_VAE_FirstLayer.DE_VAE import DE_VAE
from torch.utils.tensorboard import SummaryWriter

def loadData(path,W):
    f=open(path,'rb')
    data=pickle.load(f)
    if(W>1):
        result = np.zeros((1,3,102,W))
        for i in range(len(data)):
            data[i]=np.where(data[i]>1,1,data[i])
            data[i]=np.where(data[i]<0,0,data[i])
            result = np.vstack((result,data[i]))
        return result[1:]
    else:
        result = np.zeros((1,1))
        for i in range(len(data)):
            result = np.vstack((result,data[i]))
        return result[1:].reshape(-1,)


fresult = loadData('../RawData/FFT_AllSubject_Training_AllClass_minmaxNorm',188)
length=[x for x in range(fresult.shape[0])]
t=int(np.floor(fresult.shape[0]*0.7))
np.random.shuffle(length)
print(t)
tidxs=length[:t]
vidxs=length[t:]


de_tb = SummaryWriter('./DE_VAE_Tensorboard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
de = DE_VAE(kernelMaxW=60,kernelMinW=4,kernelMaxH=40,kernelMinH=4,bsize=1,initSize=3,trainingset=fresult[tidxs],validationset=fresult[vidxs], tb=de_tb)
start=time.time()
print(f'DE Test Run Start')
de.run()
end=time.time()
print(f'DE Test Run End runtime:{(end-start):10.8f}')

