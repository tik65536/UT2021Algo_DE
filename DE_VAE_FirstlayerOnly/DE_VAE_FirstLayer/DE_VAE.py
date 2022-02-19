#!/usr/bin/python3
import torch
import numpy as np
import pickle
#from torchsummary import summary
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import copy
import sys
from DE_VAE_FirstLayer.VAE_102x188_f8_DE import VAE


## Asssume the dim of Traing and Testing are in shape [N,C,H,W]

class DE_VAE():
    def __init__(self,kernelMaxW=10,kernelMinW=3,kernelMaxH=10,kernelMinH=3,bsize=10,epoch=100,initSize=20,maxiter=10,stopcount=3,\
                 trainingset=None,validationset=None,tb=None,outputDir='./'):
        self.best=[]
        self.mean=[]
        self.kernelMaxW=kernelMaxW
        self.kernelMinW=kernelMinW
        self.kernelMaxH=kernelMaxH
        self.kernelMinH=kernelMinH
        self.bsize = bsize
        self.epoch = epoch
        self.stopcount = stopcount
        self.pplSize = initSize
        self.maxiter = maxiter
        self.kernelSizeList = []
        self.tb = tb

        self.training = trainingset
        self.validationSet = validationset
        self.trainingSize = self.training.shape[0]
        self.channel = self.training.shape[1]
        self.H = self.training.shape[2]
        self.W = self.training.shape[3]
        H=np.random.choice(range(self.kernelMinH,self.kernelMaxH+1),self.pplSize,replace=True)
        W=np.random.choice(range(self.kernelMinW,self.kernelMaxW+1),self.pplSize,replace=True)
        self.kernelSizeList = list(zip(H,W))
        print(f"DE_VAE init ppl : {self.kernelSizeList}")
        #sys.stdout=open(outputDir+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),"w")

    def fit(self,config,id_):
        vae = VAE(config,self.channel,self.H,self.W,self.bsize)
        best = float('inf')
        stop=0
        opt = torch.optim.Adam(vae.parameters(), lr=0.001)
        loss = torch.nn.BCEWithLogitsLoss()
        batch = self.trainingSize//self.bsize
        vbatch = self.validationSet.shape[0]//self.bsize
        idxs = [x for x in range(self.training.shape[0])]
        vidxs = [x for x in range(self.validationSet.shape[0])]
        np.random.shuffle(idxs)
        for e in range(self.epoch):
            start=time.time()
            batchloss=0
            vae.train()
            for i in range(batch):
                idx=idxs[i*self.bsize:i*self.bsize+self.bsize]
                opt.zero_grad()
                data = torch.tensor(self.training[idx]).float().to(vae.device)
                out,mu,v,_ = vae(data)
                l = loss(out,data)
                KDloss = -0.5 * torch.sum(1+ v - mu.pow(2) - v.exp())
                totalloss=l+KDloss
                batchloss+=totalloss
                totalloss.backward()
                opt.step()

            vae.eval()
            np.random.shuffle(vidxs)
            vloss=0
            with torch.no_grad():
                for i in range(vbatch):
                    vidx=vidxs[i*self.bsize:i*self.bsize+self.bsize]
                    vdata = torch.tensor(self.validationSet[vidx]).float().to(vae.device)
                    vout,vmu,vv,_ = vae(vdata)
                    vl = loss(vout,vdata)
                    vKDloss = -0.5 * torch.sum(1+ vv - vmu.pow(2) - vv.exp())
                    vloss += (vl+vKDloss)
                    vout=vout.detach().cpu().numpy()
            vloss=vloss/vbatch
            if(vloss<best):
                best=vloss
            else:
                stop+=1
            end=time.time()
            print(f'DE ConfigID: {id_}, Epoch: {e:3d}, Training Loss: {(batchloss/batch):10.8f}, Validation Loss: {(vloss):10.8f},Best: {best:10.8f}, StopCount/Limit: {stop:3d}/{self.stopcount:3d}, Time:{(end-start):10.8f}')
            sys.stdout.flush()
            if(stop>=self.stopcount):
                return best,config,id_

    def mutation_rand_1_z(self,x0,x1,x2,beta):
        # number of hidden layer mutation
        #[0] is H , [1] is W
        xnew = (x0[0]+beta*(x1[0]-x2[0]),x0[1]+beta*(x1[1]-x2[1]))
        if( xnew[0] > self.kernelMaxH):
            xnew[0]=self.kernelMaxH
        if(xnew[0]<self.kernelMinH):
            xnew[0]=self.kernelMinH
        if(xnew[1]>self.kernelMaxW):
            xnew[1]=self.kernelMaxW
        if(xnew[1]<self.kernelMinW):
            xnew[1]=self.kernelMinW

    def crossoverRandomSwap(self,parent,u):
        # the first one is with min len
        swap = np.random.choice(2,2)
        if(swap[0]!=0):
           parent[0]=u[0]
        if(swap[1]!=0):
           parent[1]=u[1]
        return parent

    def run(self,beta=0.5):
        current_gen=self.kernelSizeList
        scores = np.zeros((self.pplSize))
        #initial Run
        print('DE Initial Run Start')
        sys.stdout.flush()
        for i in range(len(current_gen)):
                s,_,_ = self.fit(current_gen[i],i)
                scores[i]=s
        print('DE Initial Run End')
        sys.stdout.flush()
        currentbest = np.min(scores)
        overallBest = currentbest
        currentmean = np.mean(scores)
        currentbestidx = np.argmin(scores)
        overallBestConfig = current_gen[currentbestidx]
        bestGen = 0
        print(f'DE Init Run Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}, config: {current_gen[currentbestidx]}')
        sys.stdout.flush()
        #Generation Run
        for i in range(self.maxiter):
            updatecount=0
            start=time.time()
            print(f'DE Gen {i} Run Start')
            sys.stdout.flush()
            for j in range(self.pplSize):
                parent = current_gen[j]
                idx0,idx1,idxt = np.random.choice(range(0,self.pplSize),3,replace=False)
                unitvector = self.mutation_rand_1_z(current_gen[idxt],current_gen[idx0],current_gen[idx1],beta)
                nextGen = self.crossoverRandomSwap(parent,unitvector)
                print(f'DE Next Gen: {nextGen}')
                sys.stdout.flush()
                s,_,_=self.fit(current_gen[j],j)
                if(s<scores[j]):
                    updatecount+=1
                    scores[j]=s
                    current_gen[j]=nextGen
            print(f'DE Gen {i} Run End')
            sys.stdout.flush()
            end=time.time()
            currentbest = np.min(scores)
            currentmean = np.mean(scores)
            currentmedian = np.median(scores)
            currentq25 = np.quantile(scores,0.25)
            currentq75 = np.quantile(scores,0.75)
            currentbestidx = np.argmin(scores)
            if(currentbest<overallBest):
                overallBest=currentbest
                overallBestConfig = current_gen[currentbestidx]
                bestGen = i
            print(f'DE Run {i:3d} CurrentBest: {currentbest:10.8f}, Mean: {currentmean:10.8f}, OverallBest: {overallBest:10.8f}/{bestGen:3d}, config: {current_gen[currentbestidx]}, updatecount: {updatecount:3d}, Generation RunTime: {(end-start):10.8f}')
            sys.stdout.flush()
            if(self.tb is not None):
                self.tb.add_histogram(f'Scores',scores,i)
                self.tb.add_scalars("Scores Statistic (Generation)", {'best':currentbest,'mean':currentmean,'median':currentmedian,'q25':currentq25,'q75':currentq75 , 'OverAllBest':overallBest}, i)
                self.tb.add_scalar('Update Count',updatecount,i)
                self.tb.add_scalar('RunTime',(end-start),i)
        print(f'DE Run Completed : Best Score: {overallBest} , Config: {overallBestConfig}, find in Gen: {bestGen}')
        sys.stdout.flush()
        return


