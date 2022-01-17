import torch
import numpy as np
import pickle
#from torchsummary import summary
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import copy
import torchvision.datasets as datasets
import sys
import pymp
from DE.DNN import DNN

class SADE_MLP():
    def __init__(self, outdim=1,maxdepth=70,mindepth=5,minneuron=4,maxneuron=10,bsize=10,epoch=100,initSize=20,maxiter=10,stopcount=3,\
                 trainingset=None,validationset=None,trainingTarget=None,validateTarget=None,crossover=1,tb=None):
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
        self.tb=tb
        self.device = torch.device('cpu')
        self.training = torch.tensor(trainingset.reshape((trainingset.shape[0],-1))).float().to(self.device)
        self.validationSet = torch.tensor(validationset.reshape((validationset.shape[0],-1))).float().to(self.device)
        self.target=torch.tensor(trainingTarget).float().to(self.device)
        self.validationTarget = torch.tensor(validateTarget).float().to(self.device)

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
        bestaccuracy =0
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
                data = self.training[idx]
                y = self.target[idx]
                yhat = dnn(data)
                l = loss(yhat,y)
                batchloss+=l.item()
                l.backward()
                opt.step()

            dnn.layers.eval()
            np.random.shuffle(vidxs)
            vloss=0
            accuracy = 0
            for i in range(vbatch):
                vidx=vidxs[i*self.bsize:i*self.bsize+self.bsize]
                vdata = self.validationSet[vidx]
                vy = self.validationTarget[vidx]
                vyhat = dnn(vdata)
                vl = loss(vyhat,vy)
                vloss += vl.item()
                vyhat=vyhat.detach().numpy()
                vy=vy.detach().numpy()
                predict = np.argmax(vyhat,axis=1)
                vy = np.argmax(vy,axis=1)
                accuracy += np.where(predict==vy)[0].shape[0]
            vloss = vloss/vbatch
            accuracy = accuracy/(self.bsize*vbatch)
            if(vloss<best):
                best=vloss
                bestaccuracy = accuracy
            else:
                stop+=1
            end=time.time()
            print(f'ConfigID: {id_:3d}, Epoch: {e:3d}, Training Loss: {(batchloss/batch):10.8f}, Validation Loss: {(vloss):10.8f},Best: {best:10.8f}, Accuracy: {accuracy}, StopCount/Limit: {stop:3d}/{self.stopcount:3d}, Time:{(end-start):10.8f}')
            if(stop>=self.stopcount):
                return best,bestaccuracy,config,id_

    def mutation_rand_1_z(self,x1,xs,beta,debug=False):
        indim = x1[0]
        x1 = x1[1:-1] # remove in/out dim
        xs[0] = xs[0][1:-1]
        xs[1] = xs[1][1:-1]
        if(debug):
            print(f'M1 : x1 len {x1.shape[0]} xs0 len {xs[0].shape[0]} xs1 len {xs[1].shape[0]}')
            print(f'M1 : x1 {x1} \nM1 : xs0 {xs[0]} \nM1 : xs1 len {xs[1]}')
        #
        # A. Mutating the # of layers
        minlen = np.min([x1.shape[0],xs[0].shape[0],xs[1].shape[0]])
        if(debug): print(f'M1 : minlen {minlen}')
        newminlen = minlen
        targetlen=int(np.floor((x1.shape[0]) + beta * (xs[0].shape[0] - xs[1].shape[0])))
        if(targetlen==0): targetlen=x1.shape[0]
        elif(targetlen<0): targetlen=abs(targetlen)
        if(targetlen<self.mindepth): targetlen=self.mindepth
        if(targetlen>self.maxdepth): targetlen=self.maxdepth
        if(targetlen < minlen): newminlen=targetlen
        if(debug): print(f'M1 : New Min Len :{newminlen}, Length Mutation :{targetlen}')
        #
        # B. Mutating the # of neurons
        # As lengths of x1, xs[0], xs[1] and new length can possibly be different,
        # 1) do the mutation for # of neurons for new minlen,
        # 2) apply the same rule to remaining if needed
        xa = x1[:newminlen] + beta * (xs[0][:newminlen] - xs[1][:newminlen])
        if(targetlen>minlen):
            xaa = np.zeros((targetlen-minlen))
            a,b,c=None,None,None
            for i in range(targetlen-newminlen):
                if(x1.shape[0]<=newminlen+i): a=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(x1.shape[0]>newminlen+i): a=x1[newminlen+i]
                if(xs[0].shape[0]<=newminlen+i): b=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(xs[0].shape[0]>newminlen+i): b=xs[0][newminlen+i]
                if(xs[1].shape[0]<=newminlen+i): c=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(xs[1].shape[0]>newminlen+i): c=xs[1][newminlen+i]
                xaa[i]=a + beta * (b - c)
            xa = np.concatenate((xa, xaa), axis=None)
        for i in range(xa.shape[0]):
            if(xa[i]>self.maxneuron): xa[i]=self.maxneuron
            elif(xa[i]<self.minneuron): xa[i]=self.minneuron
            xa[i] = np.floor(xa[i])
        xa = np.concatenate((np.array(indim,dtype=int),np.array(xa,dtype=int),np.array(self.outdim,dtype=int)), axis=None,dtype=int)
        return xa

    def mutation_current2best_2_z(self,x,xp,xs,beta,debug=False):
        indim = x[0]
        x = x[1:-1]
        xp = xp[1:-1]
        xs[0] = xs[0][1:-1]
        xs[1] = xs[1][1:-1]
        if(debug):
            print(f'M1 : x len {x.shape[0]} xp len {xp.shape[0]} xs0 len {xs[0].shape[0]} xs1 len {xs[1].shape[0]}')
            print(f'M1 : x {x} \nM1 : xp {xp} \nM1 : xs0 {xs[0]} \nM1 : xs1 len {xs[1]}')
        #
        # A. Mutating the # of layers
        minlen = np.min([x.shape[0],xp.shape[0],xs[0].shape[0],xs[1].shape[0]])
        if(debug): print(f'M1 : minlen {minlen}')
        newminlen = minlen
        targetlen=int(np.floor( (x.shape[0]) + beta * (xp.shape[0] - x.shape[0]) + beta * (xs[0].shape[0] - xs[1].shape[0]) ))
        if(targetlen==0): targetlen=xp.shape[0]
        elif(targetlen<0): targetlen=abs(targetlen)
        if(targetlen<self.mindepth): targetlen=self.mindepth
        if(targetlen>self.maxdepth): targetlen=self.maxdepth
        if(targetlen < minlen): newminlen=targetlen
        if(debug): print(f'M1 : New Min Len :{newminlen}, Length Mutation :{targetlen}')
        #
        # B. Mutating the # of neurons
        # As lengths of x, xp, xs[0], xs[1] and new length can possibly be different,
        # 1) do the mutation for # of neurons for new minlen,
        # 2) apply the same rule to remaining if needed
        xa = x[:newminlen] + beta * (xp[:newminlen] - x[:newminlen]) + beta * (xs[0][:newminlen] - xs[1][:newminlen])
        if(targetlen>minlen):
            xaa = np.zeros((targetlen-minlen))
            a,b,c,d=None,None,None,None
            for i in range(targetlen-newminlen):
                if(x.shape[0]<=newminlen+i): a=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(x.shape[0]>newminlen+i): a=x[newminlen+i]
                if(xp.shape[0]<=newminlen+i): b=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(xp.shape[0]>newminlen+i): b=xp[newminlen+i]
                if(xs[0].shape[0]<=newminlen+i): c=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(xs[0].shape[0]>newminlen+i): c=xs[0][newminlen+i]
                if(xs[1].shape[0]<=newminlen+i): d=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(xs[1].shape[0]>newminlen+i): d=xs[1][newminlen+i]
                xaa[i]=a + beta * (b - a) + beta * (c - d)
            xa = np.concatenate((xa, xaa), axis=None)
        for i in range(xa.shape[0]):
            if(xa[i]>self.maxneuron): xa[i]=self.maxneuron
            elif(xa[i]<self.minneuron): xa[i]=self.minneuron
            xa[i] = np.floor(xa[i])
        xa = np.concatenate((np.array(indim,dtype=int),np.array(xa,dtype=int),np.array(self.outdim,dtype=int)), axis=None,dtype=int)
        return xa

    def crossoverUnif(self,parent,u,cr):
        r = np.random.uniform(0,1,1)[0]
        order = [parent[1:-1],u[1:-1]]
        child = [parent[0]]
        if(parent.shape[0] > u.shape[0]):
            order = [u[1:-1],parent[1:-1]]
        order[0] = np.resize(order[0],order[1].shape[0])
        jr = np.random.choice(range(order[0].shape[0]),size=1)[0]
        for j in range(order[0].shape[0]):
            if r <= cr or j == jr: child.append(order[0][j])
            else: child.append(order[1][j])
        child.append(parent[-1])
        return np.array(child)

    def run(self,mp=1):
        current_gen=self.MLPlayerlist
        scores = pymp.shared.array((self.pplSize,),dtype='float')
        accuracy = pymp.shared.array((self.pplSize,),dtype='float')
        #initial Run
        print('Initial Run Start')
        with pymp.Parallel(mp) as p:
            for i in p.range(len(current_gen)):
                b,a,_,_ = self.fit(current_gen[i],i)
                scores[i]=b
                accuracy[i]=a
        print('Initial Run End')
        currentbest = np.min(scores)
        overallBest = currentbest
        currentmean = np.mean(scores)
        currentbestidx = np.argmin(scores)
        overallBestConfig = current_gen[currentbestidx]
        overallBestAccuracy = accuracy[currentbestidx]
        currentbestAccuracy = accuracy[currentbestidx]
        bestGen = 0
        print(f'Init Run Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}, config: {current_gen[currentbestidx]}')
        # initial factors
        p1,beta,crm = 0.5,0,0.5
        crs = []
        progress = []
        #Generation Run
        for i in range(self.maxiter):
            structureStatistic=pymp.shared.array((self.pplSize,5),dtype='float')
            updatecount=0
            start=time.time()
            ns1,nf1,ns2,nf2 = 0,0,0,0
            print(f'Gen {i} Run Start')
            with pymp.Parallel(mp) as p:
                for j in p.range(self.pplSize):
                    parent = current_gen[j]
                    # factors
                    while beta <= 0 or beta > 2: beta = np.random.normal(loc=0.5,scale=0.3,size=1)[0]
                    if i%5==0: cr = np.random.normal(loc=crm,scale=0.1,size=1)[0]
                    r = np.random.uniform(low=0,high=1,size=1)[0]
                    # strategy selection + mutation
                    if r <= p1:
                        idx0,idx1,idxt = np.random.choice(range(0,self.pplSize),3,replace=False)
                        target = current_gen[idxt]
                        diff = [current_gen[idx0],current_gen[idx1]]
                        unitvector = self.mutation_rand_1_z(target,diff,beta)
                    else:
                        idxt = np.argmax(scores)
                        idx0,idx1 = np.random.choice(np.delete(np.arange(self.pplSize),idxt),2,replace=False)
                        target = current_gen[idxt]
                        diff = [current_gen[idx0],current_gen[idx1]]
                        unitvector = self.mutation_current2best_2_z(parent,target,diff,beta)
                    # crossover
                    child = self.crossoverUnif(parent,unitvector,cr)
                    print(f'Next Gen: {child}')
                    structureStatistic[j,0]= child.shape[0]
                    structureStatistic[j,1]= np.mean(child)
                    structureStatistic[j,2]= np.median(child)
                    structureStatistic[j,3]= np.quantile(child,0.25)
                    structureStatistic[j,4]= np.quantile(child,0.75)
                    # selection
                    s,a,_,_ = self.fit(child,j)
                    if(s < scores[j]):
                        updatecount+=1
                        scores[j]=s
                        accuracy[j]=a
                        current_gen[j]=child
                        crs.append(cr)
                        if r <= p1: ns1 += 1
                        else: ns2 += 1
                    else:
                        if r <= p1: nf1 += 1
                        else: nf2 += 1
            if i%5==4:
                crm = np.mean(crs)
                crs = []
            if(ns2*(ns1+nf1)+ns1*(ns2+nf2) == 0): p1 = 0
            else: p1 = (ns1*(ns2+nf2))/(ns2*(ns1+nf1)+ns1*(ns2+nf2))
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
            if(currentbest<overallBest):
                overallBest=currentbest
                overallBestConfig = current_gen[currentbestidx]
                overallBestAccuracy = accuracy[currentbestidx]
                bestGen = i
            print(f'Run {i:3d} CurrentBest: {currentbest:10.8f}, Mean: {currentmean:10.8f}, OverallBest: {overallBest:10.8f}/{bestGen:3d}, config: {current_gen[currentbestidx]}, updatecount: {updatecount:3d}, Generation RunTime: {(end-start):10.8f}')
            if(self.tb is not None):
                self.tb.add_histogram(f'Scores',scores,i)
                self.tb.add_scalars("Scores Statistic (Generation)", {'best':currentbest,'mean':currentmean,'median':currentmedian,'q25':currentq25,'q75':currentq75 , 'OverAllBest':overallBest}, i)
                self.tb.add_scalars("Structure Statistic (Generation) #HiddenLayer", {'mean':genMeanLen,'median':genMedianLen,'q25':genq25Len,'q75':genq75Len}, i)
                self.tb.add_scalars("Structure Statistic (Generation) #Node", {'mean':genMeanNode,'median':genMedianNode,'q25':genq25Node,'q75':genq75Node}, i)
                self.tb.add_scalars("Accuracy", {'Current Best':accuracy[currentbestidx],'OverallBest':overallBestAccuracy}, i)
                self.tb.add_scalar('Update Count',updatecount,i)
                self.tb.add_scalar('RunTime',(end-start),i)

        print(f'Run Completed : Best Score: {overallBest} , Config: {overallBestConfig}, find in Gen: {bestGen}')
        return
