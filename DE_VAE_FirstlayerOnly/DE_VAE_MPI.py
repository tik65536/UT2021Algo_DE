#!/usr/bin/python3
import torch
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import copy
import sys
from mpi4py import MPI
from DE_VAE_FirstLayer.VAE_102x188_f8_DE import VAE
comm = MPI.COMM_WORLD
request = MPI.Request
rank = comm.Get_rank()
size = comm.Get_size()
startTag=9898
endTag=9999
jobTag=8888

kernelMaxW=10
kernelMinW=4
kernelMaxH=10
kernelMinH=4
bsize = 1
epoch = 1
stopcount = 3
pplSize = 4
maxiter = 10
kernelSizeList = []

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


## Asssume the dim of Traing and Testing are in shape [N,C,H,W]
def fit(config,id_,rank,C,H,W,trainingset,validationset):
    vae = VAE(config,C,H,W,bsize)
    vae.share_memory()
    best = float('inf')
    stop=0
    opt = torch.optim.Adam(vae.parameters(), lr=0.001)
    loss = torch.nn.BCEWithLogitsLoss()
    batch = trainingset.shape[0]//bsize
    vbatch = validationset.shape[0]//bsize
    idxs = [x for x in range(trainingset.shape[0])]
    vidxs = [x for x in range(validationset.shape[0])]
    np.random.shuffle(idxs)
    for e in range(epoch):
        start=time.time()
        batchloss=0
        vae.train()
        for i in range(batch):
            idx=idxs[i*bsize:i*bsize+bsize]
            opt.zero_grad()
            data = (trainingset[idx]).to(vae.device)
            out,mu,v,_ = vae(data)
            l = loss(out,data)
            KDloss = -0.5 * torch.sum(1+ v - mu.pow(2) - v.exp())
            totalloss=l+KDloss
            batchloss+=totalloss.item()
            totalloss.backward()
            opt.step()

        vae.eval()
        np.random.shuffle(vidxs)
        vloss=0
        with torch.no_grad():
            for i in range(vbatch):
                vidx=vidxs[i*bsize:i*bsize+bsize]
                vdata = validationset[vidx].to(vae.device)
                vout,vmu,vv,_ = vae(vdata)
                vl = loss(vout,vdata)
                vKDloss = -0.5 * torch.sum(1+ vv - vmu.pow(2) - vv.exp())
                vloss += (vl.item()+vKDloss.item())
                vout=vout.detach().cpu().numpy()
        vloss=vloss/vbatch
        if(vloss<best):
            best=vloss
        else:
            stop+=1
        end=time.time()
        print(f'Rank {rank} DE Config: {config}, Epoch: {e:3d}, Training Loss: {(batchloss/batch):10.8f}, Validation Loss: {(vloss):10.8f},Best: {best:10.8f}, StopCount/Limit: {stop:3d}/{stopcount:3d}, Time:{(end-start):10.8f}')
        if(stop>=stopcount):
            return best
    return best

def crossoverRandomSwap(self,parent,u):
    # the first one is with min len
    swap = np.random.choice(2,2)
    if(swap[0]!=0):
       parent[0]=u[0]
    if(swap[1]!=0):
       parent[1]=u[1]
    return parent

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


def run(beta=0.5):
    current_gen=kernelSizeList
    scores = np.zeros((pplSize))
    jobidx=[ -1 for i in range(size)]
    jobReq=[]
    jobreqBuf = np.empty((size,1))
    resultReq=[]
    resultBuf= np.empty((size,1),dtype=float)
    Jobdone=False
    head=0
    #initial Run
    print(f'Rank {rank} DE Initial Run Start')
    for i in range(size):
        jobReq.append(comm.Irecv(jobreqBuf[i],source=i,tag=startTag))
    while(not Jobdone):
        status = [ MPI.Status() for i in range(size) ]
        reqJobResponse = request.Testsome(jobReq,status)
        if(reqJobResponse is not None and  len(reqJobResponse)>0):
            for i in status:
                if(head<pplSize and i.source>0):
                    src = i.source
                    print(f"Rank {rank} Job Req received from: {src}")
                    buf=np.array(list(current_gen[head]),dtype=int)
                    print(f"Rank {rank} send Job Details ksize: {buf}")
                    req=comm.Isend(buf.copy(),src,tag=jobTag)
                    req.wait()
                    jobidx[src]=head
                    tmp=np.zeros(1)
                    resultReq.append(comm.Irecv(tmp,source=src,tag=jobTag))
                    resultBuf[src]= tmp
                    head+=1
                    jobReq[src]=comm.Irecv(jobreqBuf[src],tag=startTag)
        status = [ MPI.Status() for i in range(size) ]
        reqResultResponse = request.Testsome(resultReq,status)
        if(reqResultResponse is not None and  len(reqResultResponse)>0):
            for i in status:
                if(i.source>0):
                    src = i.source
                    result = resultBuf[src].copy()
                    print(f'Rank {rank} received Result from {src} : {result}')
                    idx = jobidx[src]
                    scores[idx]=result
        if(request.Waitall(resultReq) and head==pplSize ):
            Jobdone=True
    print(f'Rank {rank} DE Initial Run End')
    currentbest = np.min(scores)
    overallBest = currentbest
    currentmean = np.mean(scores)
    currentbestidx = np.argmin(scores)
    overallBestConfig = current_gen[currentbestidx]
    bestGen = 0
    print(f'Rank {rank} DE Init Run Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}, config: {current_gen[currentbestidx]}')
    #Generation Run
    for i in range(maxiter):
        updatecount=0
        start=time.time()
        print(f'Rank {rank} DE Gen {i} Run Start')
        jobidx=[ -1 for i in range(size)]
        jobReq=[]
        jobreqBuf = np.empty((size,1))
        resultReq=[]
        resultBuf= np.empty((size,1),dtype=float)
        nextGenlst = [() for i in range(size) ]
        Jobdone=False
        head=0
        for i in range(size):
            jobReq.append(comm.Irecv(jobreqBuf[i],source=i,tag=startTag))
        while(not Jobdone):
            status = [ MPI.Status() for i in range(size) ]
            reqJobResponse = request.Testsome(jobReq,status)
            print(reqJobResponse)
            if( reqJobResponse is not None and  len(reqJobResponse)>0):
                for i in status:
                     if(head<pplSize and i.source > 0):
                        src = status[i].source
                        print(f"Rank {rank} Job Req received from: {src}")
                        parent = current_gen[head]
                        idx0,idx1,idxt = np.random.choice(range(0,pplSize),3,replace=False)
                        unitvector = mutation_rand_1_z(current_gen[idxt],current_gen[idx0],current_gen[idx1],beta)
                        nextGen = crossoverRandomSwap(parent,unitvector)
                        print(f'Rank {rank} DE Next Gen: {nextGen}')
                        buf=np.array(list(nextGen),dtype=int)
                        print(f"Rank {rank} send Job Details ksize: {buf}")
                        req=comm.Isend(buf,src,tag=jobTag)
                        nextGenlst[src]=nextGen
                        jobidx[src]=head
                        req.wait()
                        tmp=np.zeros(1)
                        resultReq.append(comm.Irecv(tmp,source=src,tag=jobTag))
                        resultBuf[src]= tmp
                        head+=1
                        jobReq[src]=comm.Irecv(jobreqBuf[src],tag=startTag)
            status = [ MPI.Status() for i in range(size) ]
            reqResultResponse = request.Testsome(resultReq,status)
            print(reqResultResponse)
            if(reqResultResponse is not None and len(reqResultResponse)>0):
                for i in status:
                    if(i.source > 0):
                        src = status[i].source
                        result = resultBuf[src].copy()
                        print(f'Rank {rank} received Result from {src} : {result}')
                        idx = jobidx[src]
                        if(result<scores[idx]):
                            updatecount+=1
                            scores[idx]=result
                            current_gen[idx]=nextGenlst[src]
            if(request.Waitall(resultReq) and head==pplSize ):
                Jobdone=True
        print(f'Rank {rank} DE Gen {i} Run End')
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
        print(f'Rank {rank} DE Run {i:3d} CurrentBest: {currentbest:10.8f}, Mean: {currentmean:10.8f}, OverallBest: {overallBest:10.8f}/{bestGen:3d}, config: {current_gen[currentbestidx]}, updatecount: {updatecount:3d}, Generation RunTime: {(end-start):10.8f}')
        #if(tb is not None):
        #    tb.add_histogram(f'Scores',scores,i)
        #    tb.add_scalars("Scores Statistic (Generation)", {'best':currentbest,'mean':currentmean,'median':currentmedian,'q25':currentq25,'q75':currentq75 , 'OverAllBest':overallBest}, i)
        #    tb.add_scalar('Update Count',updatecount,i)
        #    tb.add_scalar('RunTime',(end-start),i)
    print(f'Rank {rank} DE Run Completed : Best Score: {overallBest} , Config: {overallBestConfig}, find in Gen: {bestGen}')
    return

def runWorker(C,H,W,trainingset,validationset):
    count=0
    while(True):
        comm.Isend(np.empty(1),0,tag=startTag)
        buf = np.zeros(2,dtype=int)
        comm.Recv(buf,source=0,tag=jobTag)
        #print(f'\tRank {rank} - Wait for job')
        print(f'{rank} received ksize: {buf}')
        r = fit((int(buf[0]),int(buf[1])),0,rank,C,H,W,trainingset,validationset)
        req=comm.Isend(np.array([r],dtype=float),0,tag=jobTag)
        req.wait()
        count+=1


fresult = loadData('../RawData/FFT_AllSubject_Training_AllClass_minmaxNorm',188)
training = torch.tensor(fresult).float()
C = training.shape[1]
H = training.shape[2]
W = training.shape[3]

length=[x for x in range(fresult.shape[0])]
t=int(np.floor(fresult.shape[0]*0.7))
np.random.shuffle(length)
tidxs=length[:t]
vidxs=length[t:]

if(rank==0):
    H=np.random.choice(range(kernelMinH,kernelMaxH+1),pplSize,replace=True)
    W=np.random.choice(range(kernelMinW,kernelMaxW+1),pplSize,replace=True)
    kernelSizeList = list(zip(H,W))
    print(f"DE_VAE init ppl : {kernelSizeList}")
    run()
else:
    runWorker(C,H,W,training[tidxs],training[vidxs])


#sys.stdout=open(outputDir+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),"w")
