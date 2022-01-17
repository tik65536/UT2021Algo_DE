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
sys.path.append('../Utils')
from DE.DNN import DNN


## Asssume the dim of Traing and Testing are in shape [N,C,H,W]

class JADE_MLP():
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
        self.MLPlayerlist = []
        self.depthlist=np.random.choice(range(self.mindepth,self.maxdepth),self.pplSize,replace=True)
        self.crossover=crossover
        self.adap_conf = (0.1,0.1,0.1,0.9)
        self.tb = tb
        #if torch.cuda.is_available():
        #    self.device = torch.device('cuda')
        #else:
        self.device = torch.device('cpu')
        self.training = torch.tensor(trainingset.reshape((trainingset.shape[0],-1))).float().to(self.device)
        self.validationSet = torch.tensor(validationset.reshape((validationset.shape[0],-1))).float().to(self.device)
        self.target=torch.tensor(trainingTarget).float().to(self.device)
        self.validationTarget = torch.tensor(validateTarget).float().to(self.device)

        # Generate initial population
        for i in range(self.pplSize):
            depth = self.depthlist[i]
            tmp = []
            # the number of neurons for the first layer is the dimension of the element in training data (in our case the size of the image)
            tmp.append(self.training.shape[1])
            for j in range(depth):
                # generate the number of neurons for each layer
                tmp.append(np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0])
            tmp.append(self.outdim) # last layer consist of 1 neuron by default
            tmp=np.array(tmp)
            self.MLPlayerlist.append(tmp)

    # define fit function - it calculates the fitness of one individual (one NN)
    def fit(self,config,id_,p=None):
        dnn = DNN(config) # define DNN based on configurations (layers and neurons)
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
            # training
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
            # validating
            dnn.layers.eval()
            np.random.shuffle(vidxs)
            vloss=0
            accuracy = 0
            for i in range(vbatch):
                accuracy = 0
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
            # updating best loss
            if(vloss<best):
                best=vloss
                bestaccuracy = accuracy
            # updating stopping condition
            else: stop+=1
            end=time.time()
            print(f'JADE ConfigID: {id_:3d}, Epoch: {e:3d}, Training Loss: {(batchloss/batch):10.8f}, Validation Loss: {(vloss):10.8f},Best: {best:10.8f}, Accuracy: {accuracy:4.4f}, StopCount/Limit: {stop:3d}/{self.stopcount:3d}, Time:{(end-start):10.8f}')
            # stopping condition and stopping
            if(stop>=self.stopcount):
                return best,bestaccuracy,config,id_

    def jde_params(self,beta,cr):
        tau1,tau2,beta1,betau = self.adap_conf
        r1,r2,r3,r4 = np.random.uniform(0,1,4)
        if(r2 < tau1): beta = round(beta1 + r1 * betau,3) # else, keep the beta same
        if(r4 < tau2): cr = r3
        return beta,cr

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
        # check the sign of targetlen: if the new length == 0 , set it back to target len , if <0 , take abs
        if(targetlen==0): targetlen=x1.shape[0]
        elif(targetlen<0): targetlen=abs(targetlen)
        # check if new length is between mindepth and maxdepth
        if(targetlen < self.mindepth): targetlen = self.mindepth
        elif(targetlen > self.maxdepth): targetlen = self.maxdepth
        # new minimum length is min of minlen and targetlen
        if(targetlen < minlen): newminlen=targetlen
        if(debug): print(f'M1 : New Min Len :{newminlen}, Length Mutation :{targetlen}')
        #
        # B. Mutating the # of neurons
        # As lengths of x1, xs[0], xs[1] and new length can possibly be different,
        # 1) do the mutation for # of neurons for new minlen,
        # 2) apply the same rule to remaining if needed
        # xa = np.zeros((targetlen),dtype=int)
        # Mutating the number of neurons up to min len layers
        xa = x1[:newminlen] + beta * (xs[0][:newminlen] - xs[1][:newminlen]) # mutate on node with minlen
        # Mutating the number of neurons for the rest layers
        if(targetlen>minlen):
            xaa = np.zeros((targetlen-minlen))
            a,b,c=None,None,None
            for i in range(targetlen-newminlen): # if number of neurons missing in vector, generate random from range (min)
                if(x1.shape[0]<=newminlen+i): a=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(x1.shape[0]>newminlen+i): a=x1[newminlen+i]
                if(xs[0].shape[0]<=newminlen+i): b=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(xs[0].shape[0]>newminlen+i): b=xs[0][newminlen+i]
                if(xs[1].shape[0]<=newminlen+i): c=np.random.choice(range(self.minneuron,self.maxneuron),1,replace=False)[0]
                elif(xs[1].shape[0]>newminlen+i): c=xs[1][newminlen+i]
                xaa[i]=a + beta * (b - c)
            xa = np.concatenate((xa, xaa), axis=None)
        # check if numbers of neurons are in allowed range
        for i in range(xa.shape[0]):
            if(xa[i]>self.maxneuron): xa[i]=self.maxneuron
            elif(xa[i]<self.minneuron): xa[i]=self.minneuron
            xa[i] = np.floor(xa[i])
        xa = np.concatenate((np.array(indim,dtype=int),np.array(xa,dtype=int),np.array(self.outdim,dtype=int)), axis=None,dtype=int)
        return xa

    def crossoverMean(self,parent,u):
        order = [parent[1:-1],u[1:-1]]
        if(parent.shape[0] > u.shape[0]): order = [u[1:-1],parent[1:-1]]
        order[0] = np.resize(order[0],order[1].shape[0])
        middle = np.mean(order,axis=0,dtype=int)
        child=np.insert(middle,0,parent[0])
        child=np.append(child,parent[-1])
        return child.copy()

    def crossoverRandomSwap(self,parent,u):
        # the first one is with min len
        order = [parent[1:-1],u[1:-1]]
        child = [parent[0]]
        if(parent.shape[0] > u.shape[0]): order = [u[1:-1],parent[1:-1]]
        order[0] = np.resize(order[0],order[1].shape[0])
        swap = np.random.randint(0,2,order[0].shape[0])
        for i in range(len(swap)):
            if(swap[i]==0): child.append(order[0][i])
            else: child.append(order[1][i])
        child.append(parent[-1])
        return np.array(child).copy()

    def crossoverJDESwap(self,parent,u,cr):
        # the first one is with min len
        order = [parent[1:-1],u[1:-1]]
        child = [parent[0]]
        if(parent.shape[0] > u.shape[0]): order = [u[1:-1],parent[1:-1]]
        order[0] = np.resize(order[0],order[1].shape[0])
        swap = np.random.randint(0,2,order[0].shape[0])
        for i in range(len(swap)):
            r = np.random.uniform(0,1,1)[0]
            if(swap[i]==0 or r<=cr): child.append(order[0][i])
            else: child.append(order[1][i])
        child.append(parent[-1])
        return np.array(child).copy()

    def run(self,beta=0.5,cr=0.9):
        current_gen=self.MLPlayerlist
        scores = np.zeros((self.pplSize))
        accuracy = np.zeros((self.pplSize))
        #initial Run
        print('JADE Initial Run Start')
        for i in range(len(self.MLPlayerlist)):
            b,a,_,_ = self.fit(self.MLPlayerlist[i],i)
            scores[i]=b
            accuracy[i]=a
        print('JADE Initial Run End')
        currentbest = np.min(scores)
        overallBest = currentbest
        currentmean = np.mean(scores)
        currentbestidx = np.argmin(scores)
        overallBestConfig = current_gen[currentbestidx]
        overallBestAccuracy = accuracy[currentbestidx]
        currentbestAccuracy = accuracy[currentbestidx]
        bestGen = 0
        print(f'JADE Init Run Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}, config: {current_gen[currentbestidx]}')
        #Generation Run
        for i in range(self.maxiter):
            structureStatistic=np.zeros((self.pplSize,5))
            updatecount=0
            start=time.time()
            print(f'JADE Gen {i} Run Start')
            betas = np.ones(self.pplSize)*beta
            crs = np.ones(self.pplSize)*cr
            for j in range(self.pplSize):
                parent = current_gen[j]
                idx0,idx1,idxt = np.random.choice(range(0,self.pplSize),3,replace=False)
                target = current_gen[idxt]
                diff = [current_gen[idx0],current_gen[idx1]]
                betas[j],crs[j] = self.jde_params(betas[j],crs[j])
                unitvector = self.mutation_rand_1_z(target,diff,betas[j])
                nextGen = self.crossoverJDESwap(parent,unitvector,crs[j])
                print(f'JADE Next Gen: {nextGen}')
                structureStatistic[j,0]= nextGen.shape[0]-2
                structureStatistic[j,1]= np.mean(nextGen[1:-1])
                structureStatistic[j,2]= np.median(nextGen[1:-1])
                structureStatistic[j,3]= np.quantile(nextGen[1:-1],0.25)
                structureStatistic[j,4]= np.quantile(nextGen[1:-1],0.75)
                s,a,_,_ = self.fit(nextGen,j)
                if(s<scores[j]):
                    updatecount+=1
                    scores[j]=s
                    accuracy[j]=a
                    current_gen[j]=nextGen
            print(f'JADE Gen {i} Run End')
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
            print(f'JADE Run {i:3d} CurrentBest: {currentbest:10.8f}, Mean: {currentmean:10.8f}, OverallBest: {overallBest:10.8f}/{bestGen:3d}, config: {current_gen[currentbestidx]}, updatecount: {updatecount:3d}, Generation RunTime: {(end-start):10.8f}')
            if(self.tb is not None):
                self.tb.add_histogram(f'Scores',scores,i)
                self.tb.add_scalars("Scores Statistic (Generation)", {'best':currentbest,'mean':currentmean,'median':currentmedian,'q25':currentq25,'q75':currentq75 , 'OverAllBest':overallBest}, i)
                self.tb.add_scalars("Structure Statistic (Generation) #HiddenLayer", {'mean':genMeanLen,'median':genMedianLen,'q25':genq25Len,'q75':genq75Len}, i)
                self.tb.add_scalars("Structure Statistic (Generation) #Node", {'mean':genMeanNode,'median':genMedianNode,'q25':genq25Node,'q75':genq75Node}, i)
                self.tb.add_scalars("Accuracy", {'Current Best':accuracy[currentbestidx],'OverallBest':overallBestAccuracy}, i)
                self.tb.add_scalar('Update Count',updatecount,i)
                self.tb.add_scalar('RunTime',(end-start),i)
        print(f'JADE Run Completed : Best Score: {overallBest} , Config: {overallBestConfig}, find in Gen: {bestGen}')
        return

