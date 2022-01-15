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
import pymp

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers)-1
        #print(layers)
        # set up layer order dict
        self.activation = torch.nn.ReLU

        layer_list = list()
        for i in range(self.depth-1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        #print(layers[-2])
        #print(layers[-1])

        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

## Asssume the dim of Traing and Testing are in shape [N,C,H,W]

class DE_MLP():
    def __init__(self, outdim=1,maxdepth=70,mindepth=5,minneuron=4,maxneuron=10,bsize=10,epoch=100,initSize=20,maxiter=10,stopcount=3,\
                 trainingset=None,validationset=None,trainingTarget=None,validateTarget=None,crossover=1):
        self.best=[]
        self.mean=[]
        self.outdim=outdim
        self.maxdepth=maxdepth
        self.mindepth=mindepth
        self.minneuron = minneuron
        self.maxneuron = maxneuron
        self.bsize = bsize
        self.epoch = epoch
        self.stopcount = stopcount
        self.pplSize = initSize
        self.maxiter = maxiter
        self.training = trainingset.reshape((trainingset.shape[0],-1))
        self.validationSet = validationset.reshape((validationset.shape[0],-1))
        self.target=trainingTarget
        self.validationTarget = validateTarget
        self.MLPlayerlist = []
        self.depthlist=np.random.choice(range(self.mindepth,self.maxdepth),self.pplSize,replace=True)
        self.crossover=crossover
        self.tb = SummaryWriter('./Statistic/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        #if torch.cuda.is_available():
        #    self.device = torch.device('cuda')
        #else:
        self.device = torch.device('cpu')

        for i in range(self.pplSize):
            depth = self.depthlist[i]
            tmp=[]
            tmp.append(self.training.shape[1])
            for j in range(depth):
                tmp.append(np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0])
            tmp.append(self.outdim)
            tmp=np.array(tmp)
            self.MLPlayerlist.append(tmp)

    def fit(self,config,id_,p=None):
        dnn = DNN(config)
        dnn.layers.to(self.device)
        best = float('inf')
        stop=0
        opt = torch.optim.Adam(dnn.layers.parameters(), lr=0.001)
        loss = torch.nn.BCEWithLogitsLoss()
        batch = self.training.shape[0]//self.bsize
        vbatch = self.validationSet.shape[0]//self.bsize
        idxs = [x for x in range(self.training.shape[0])]
        vidxs = [x for x in range(self.validationSet.shape[0])]
        for e in range(self.epoch):
            start=time.time()
            np.random.shuffle(idxs)
            dnn.layers.train()
            batchloss=0
            for i in range(batch):
                idx=idxs[i*self.bsize:i*self.bsize+self.bsize]
                opt.zero_grad()
                data = torch.tensor(self.training[idx]).float().to(self.device)
                y = torch.tensor(self.target[idx]).float().to(self.device)
                yhat = dnn(data)
                l = loss(yhat,y)
                batchloss+=l.item()
                l.backward()
                opt.step()

            dnn.layers.eval()
            np.random.shuffle(vidxs)
            vloss=0
            for i in range(vbatch):
                vidx=vidxs[i*self.bsize:i*self.bsize+self.bsize]
                vdata = torch.tensor(self.validationSet[vidx]).float().to(self.device)
                vy = torch.tensor(self.validationTarget[vidx]).float().to(self.device)
                vyhat = dnn(vdata)
                vl = loss(vyhat,vy)
                vloss += vl.item()
            vloss=vloss/vbatch
            if(vloss<best):
                best=vloss
            else:
                stop+=1
            end=time.time()
            if(p is not None):
                print(f'Process: {p:3d}, ConfigID: {id_:3d}, Epoch: {e:3d}, Training Loss: {(batchloss/batch):10.8f}, Validation Loss: {(vloss/vbatch):10.8f},\
                    Best: {best/vbatch:10.8f}, StopCount/Limit: {stop:3d}/{self.stopcount:3d}, Time:{(end-start):10.8f}')
            else:
                print(f'ConfigID: {id_:3d}, Epoch: {e:3d}, Training Loss: {(batchloss/batch):10.8f}, Validation Loss: {(vloss/vbatch):10.8f},\
                    Best: {best/vbatch:10.8f}, StopCount/Limit: {stop:3d}/{self.stopcount:3d}, Time:{(end-start):10.8f}')
            if(stop>=self.stopcount):
                return best,config,id_

    def mutation_rand_1(self,x1,xs,beta,debug=False):
        indim = x1[0]
        x1 = x1[1:-1] # remove in/out dim
        xs[0] = xs[0][1:-1]
        xs[1] = xs[1][1:-1]
        if(debug):
            print(f'M1 : x1 len {x1.shape[0]} xs0 len {xs[0].shape[0]} xs1 len {xs[1].shape[0]}')
            print(f'M1 : x1 {x1} \nM1 : xs0 {xs[0]} \nM1 : xs1 len {xs[1]}')
        # number of hidden layer mutation
        # if the new length == 0 , set it back to target len , if <0 , take abs
        minlen = np.min([x1.shape[0],xs[0].shape[0],xs[1].shape[0]])
        if(debug):
            print(f'M1 : minlen {minlen}')
        newminlen = minlen
        targetlen=int(np.floor((x1.shape[0]) + beta * (xs[0].shape[0] - xs[1].shape[0])))
        if(targetlen==0):
            targetlen=x1.shape[0]
        elif(targetlen<0):
            targetlen=abs(targetlen)
        if(targetlen<self.mindepth):
            targetlen=self.mindepth
        if(targetlen>self.maxdepth):
            targetlen=self.maxdepth
        if(targetlen < minlen):
            newminlen=targetlen
        if(debug):
            print(f'M1 : New Min Len :{newminlen}, Length Mutation :{targetlen}')

        #node mutation
        #As x1'len , xs' len and new len will be different, it will first do the node number mutation for min len, then apply the same rule to remaining if need.
        # Node number mutation up to min len
        xa = x1[:newminlen] + beta * (xs[0][:newminlen] - xs[1][:newminlen]) # mutate on node with minlen
        # Node number mutation for the rest
        if(targetlen>minlen):
            xaa = np.zeros((targetlen-minlen))
            a=None
            b=None
            c=None
            for i in range(targetlen-newminlen):
                if(x1.shape[0]<=newminlen+i):
                    a=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                if(x1.shape[0]>newminlen+i):
                    a=x1[newminlen+i]

                if(xs[0].shape[0]<=newminlen+i):
                    b=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(xs[0].shape[0]>newminlen+i):
                    b=xs[0][newminlen+i]

                if(xs[1].shape[0]<=newminlen+i):
                    c=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(xs[1].shape[0]>newminlen+i):
                    c=xs[1][newminlen+i]

                xaa[i]=a + beta * (b - c)

            xa = np.concatenate((xa, xaa), axis=None)

        for i in range(xa.shape[0]):
            if(xa[i]>self.maxneuron):
                 xa[i]=self.maxneuron
            elif(xa[i]<self.minneuron):
                  xa[i]=self.minneuron
            xa[i] = np.floor(xa[i])
        xa = np.concatenate((np.array(indim,dtype=int),np.array(xa,dtype=int),np.array(self.outdim,dtype=int)), axis=None,dtype=int)
        return xa

    def crossoverMean(self,parent,u):
        order = [parent[1:-1],u[1:-1]]
        if(parent.shape[0] > u.shape[0]):
            order = [u[1:-1],parent[1:-1]]
        order[0] = np.resize(order[0],order[1].shape[0])
        middle = np.mean(order,axis=0,dtype=int)
        child=np.insert(middle,0,parent[0])
        child=np.append(child,parent[-1])
        return child.copy()

    def crossoverRandomSwap(self,parent,u):
        # the first one is with min len
        order = [parent[1:-1],u[1:-1]]
        child = [parent[0]]
        if(parent.shape[0] > u.shape[0]):
            order = [u[1:-1],parent[1:-1]]
        order[0] = np.resize(order[0],order[1].shape[0])
        swap = np.random.randint(0,2,order[0].shape[0])

        for i in range(len(swap)):
            if(swap[i]==0):
                child.append(order[0][i])
            else:
                child.append(order[1][i])
        child.append(parent[-1])
        return np.array(child).copy()

    def run(self,beta=0.5):
        current_gen=self.MLPlayerlist
        scores = np.zeros((self.pplSize))
        #initial Run
        print('Initial Run Start')
        for i in range(len(current_gen)):
            b,_,_ = self.fit(current_gen[i],i)
            scores[i]=b
        print('Initial Run End')
        currentbest = np.min(scores)
        currentmean = np.mean(scores)
        currentbestidx = np.argmin(scores)
        print(f'Init Run Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}, config: {current_gen[currentbestidx]}')
        #Generation Run
        for i in range(self.maxiter):
            structureStatistic=np.zeros((self.pplSize,5))
            updatecount=0
            start=time.time()
            print(f'Gen {i} Run Start')
            for j in range(self.pplSize):
                parent = current_gen[j]
                midxs = np.random.choice(range(0,self.pplSize),3,replace=False)
                target = current_gen[midxs[2]]
                unitvector = self.mutation_rand_1(target,current_gen[midxs[0:2]],beta)
                #print(f'U: {unitvector}')
                #print(f'P: {parent}')
                if(self.crossover==1):
                    nextGen = self.crossoverMean(parent,unitvector)
                else:
                    nextGen = self.crossoverRandomSwap(parent,unitvector)
                print(f'Next Gen: {nextGen}')
                structureStatistic[j,0]= nextGen.shape[0]
                structureStatistic[j,1]= np.mean(nextGen)
                structureStatistic[j,2]= np.median(nextGen)
                structureStatistic[j,3]= np.quantile(nextGen,0.25)
                structureStatistic[j,4]= np.quantile(nextGen,0.75)
                s,_,_ = self.fit(nextGen,j)
                if(s<scores[j]):
                    updatecount+=1
                    scores[j]=s
                    current_gen[j]=nextGen
            print(f'Gen {i} Run End')
            end=time.time()
            currentbest = np.min(scores)
            currentmean = np.mean(scores)
            currentmedian = np.median(scores)
            currentq25 = np.quantile(scores,0.25)
            currentq75 = np.quantile(scores,0.75)
            currentbestidx = np.argmin(scores)
            genMeanLen = np.mean(structureStatistic[:,0])
            genMedianLen = np.median(structureStatistic[:,0])
            genq25Len = np.quantile(structureStatistic[:,0],0.25)
            genq75Len = np.quantile(structureStatistic[:,0],0.75)
            genMeanNode=np.median(structureStatistic[:,1])
            genMedianNode=np.median(structureStatistic[:,2])
            genq25Node = np.median(structureStatistic[:,3])
            genq75Node = np.median(structureStatistic[:,4])
            print(f'Run {i:3d} Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}, config: {current_gen[currentbestidx]}, updatecount: {updatecount:3d}, Generation RunTime: {(end-start):10.8f}')
            self.tb.add_scalars("Scores Statistic (Generation)", {'best':currentbest,'mean':currentmean,'median':currentmedian,'q25':currentq25,'q75':currentq75}, i)
            self.tb.add_scalars("Structure Statistic (Generation) #HiddenLayer", {'mean':genMeanLen,'median':genMedianLen,'q25':genq25Len,'q75':genq75Len}, i)
            self.tb.add_scalars("Structure Statistic (Generation) #Node", {'mean':genMeanNode,'median':genMedianNode,'q25':genq25Node,'q75':genq75Node}, i)
            self.tb.add_scalar('Update Count',updatecount,i)
            self.tb.add_scalar('RunTime',(end-start),i)
        print(f'Run Completed : Best Score(loss): {np.min(scores)} , Config: {current_gen[np.argmin(scores)]}')
        return


