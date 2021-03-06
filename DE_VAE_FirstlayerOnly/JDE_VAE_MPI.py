#!/usr/bin/python3
import torch
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from mpi4py import MPI
from VAE_Model.VAE_102x188_f8_DE import VAE
import sharearray
comm = MPI.COMM_WORLD
request = MPI.Request
rank = comm.Get_rank()
size = comm.Get_size()
startTag=9898
waitTag=9999
jobTag=8888


kernelMaxW=60
kernelMinW=4
kernelMaxH=40
kernelMinH=4
bsize = 100
epoch = 100
stopcount = 3
pplSize = 21
maxiter = 10
kernelSizeList = []
sleep=60
dataPath = './RawData/FFT_AllSubject_Training_AllClass_minmaxNorm'
datawidth=188
adaptivePara=[0.1,0.1,0.1,0.9]
tb=SummaryWriter("./Tensorboard_/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_MaxH'+str(kernelMaxH)+'_MaxW'+str(kernelMaxW)+'_MinH'+str(kernelMinH)+'_MinW'+str(kernelMinW))

@sharearray.decorator('CNSFFTDATA', verbose=False)
def loadData():
    global dataPath
    global datawidth
    f=open(dataPath,'rb')
    data=pickle.load(f)
    if(datawidth>1):
        result = np.zeros((1,3,102,datawidth))
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

fresult = loadData()
training = torch.tensor(fresult).float()

## Asssume the dim of Traing and Testing are in shape [N,C,H,W]
def fit(config,id_,rank,C,H,W,trainidx,validateidx):
    vae = VAE(config,C,H,W,bsize)
    best = float('inf')
    stop=0
    trainingset = training[trainidx]
    validationset = training[validateidx]
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

def jde_params(self,beta,cr):
        tau1,tau2,beta1,betau = adaptivePara
        r1,r2,r3,r4 = np.random.uniform(0,1,4)
        if(r2 < tau1): beta = round(beta1 + r1 * betau,3) # else, keep the beta same
        if(r4 < tau2): cr = r3
        return beta,cr

def crossoverRandomSwap(parent,u,cr):
    # the first one is with min len
    r = np.random.uniform(0,1,2)
    if(r[0]<=cr):
       parent[0]=u[0]
    if(r[1]>=cr):
       parent[1]=u[1]
    return parent

def mutation_rand_1_z(x0,x1,x2,beta):
    # number of hidden layer mutation
    #[0] is H , [1] is W
    xnew = [int(x0[0]+beta*(x1[0]-x2[0])),int(x0[1]+beta*(x1[1]-x2[1]))]
    if( xnew[0] > kernelMaxH):
        xnew[0]=kernelMaxH
    if(xnew[0]<kernelMinH):
        xnew[0]=kernelMinH
    if(xnew[1]>kernelMaxW):
        xnew[1]=kernelMaxW
    if(xnew[1]<kernelMinW):
        xnew[1]=kernelMinW
    return xnew



def run(beta=0.5,cr=0.9):
    current_gen=kernelSizeList
    scores = np.full((pplSize),np.inf,dtype=float)
    jobreqBuf = [ np.full(1,-1,dtype=int) for x in range(1,size) ]
    jobReq = [ comm.Irecv(jobreqBuf[x-1],source=x,tag=startTag) for x in range(1,size)]
    overallBest=float('inf')
    overallBestConfig=None
    bestGen=0
    #initial Run
    for r in range(maxiter+1):
        print(f'Rank {rank} DE Run {r} Start')
        print(f"DE_VAE Current Gen{r:3d} : {current_gen}")
        #resultBuf = [ np.zeros((5),dtype=float) for x in range(1,size) ]
        #resultReq = [ MPI.Request() for x in range(1,size) ]
        resultBuf = []
        resultReq = []
        betalist = np.ones(pplSize)*beta
        crlist = np.ones(pplSize)*cr
        joblist=np.full(pplSize,1,dtype=int)
        Jobdone=False
        head=0
        updatecount=0
        start = time.time()
        counter=0
        while(not Jobdone):
            status = [MPI.Status() for x in range(len(jobReq))]
            try:
                reqJobResponse = request.Testsome(jobReq,status)
            except:
                print(f'Rank {rank} Debug jobReq Status Exception : {[MPI.Get_error_string(i.error) for i in status]}')
            if(reqJobResponse is not None and  len(reqJobResponse)>0):
                print(f'Rank {rank} Debug reqJobResponse :{reqJobResponse}')
                for i in reqJobResponse:
                    if(head<pplSize):
                        jobReq[i].wait()
                        print("Rank {rank} Debug Wait jobReq Finished")
                        src = jobreqBuf[i][0]
                        print(f"Rank {rank} Job Req received from: {src}")
                        nextGen = current_gen[head].copy()
                        if(r>0):
                            betalist[head],crlist[head] = jde_params(betalist[head],crlist[head])
                            idx0,idx1,idxt = np.random.choice(range(0,pplSize),3,replace=False)
                            unitvector = mutation_rand_1_z(current_gen[idxt],current_gen[idx0],current_gen[idx1],betalist[head])
                            nextGen = crossoverRandomSwap(nextGen,unitvector,crlist[head])
                        print(f'Rank {rank} DE Next Gen: {nextGen}')
                        buf=np.append(np.array(nextGen,dtype=int),head)
                        print(f"Rank {rank} send Job({head}) Details ksize: {buf}")
                        req=comm.Isend(buf.copy(),src,tag=jobTag)
                        reqBufTmp=np.zeros((5),dtype=float)
                        resultReq.append(comm.Irecv(reqBufTmp,src,tag=jobTag))
                        resultBuf.append(reqBufTmp)
                        head+=1
                        req.wait()
                        print("Rank {rank} Debug Wait resultReq Send Finished")
                    elif(head>=pplSize):
                        jobReq[i].wait()
                        print("Rank {rank} Debug Wait jobReq NoMoreJob Finished")
                        src = int(jobreqBuf[i][0])
                        print(f"Rank {rank} All job is sent out , set to Wait:{src}")
                        comm.Isend(np.zeros(1),src,tag=waitTag)
                        jobreqBuftmp=np.full(1,-1,dtype=int)
                        jobReq.append(comm.Irecv(jobreqBuftmp,src,tag=startTag))
                        jobreqBuf.append(jobreqBuftmp)

            status = [ MPI.Status() for i in range(len(resultReq)) ]
            try:
                reqResultResponse = request.Testsome(resultReq,status)
            except:
                print(f'Rank {rank} Debug resultReq Status Exception : {[MPI.Get_error_string(i.error) for i in status]}')
            if(reqResultResponse is not None and  len(reqResultResponse)>0):
                print(f'Debug reqResultResponse {reqResultResponse}')
                for j in reqResultResponse:
                    print("Rank {rank} Debug Wait resultReq Receive Start")
                    resultReq[j].wait()
                    print("Rank {rank} Debug Wait resultReq Receive Finished")
                    result = resultBuf[j].copy()
                    src = int(result[4])
                    jobreqBuftmp=np.full(1,-1,dtype=int)
                    jobReq.append(comm.Irecv(jobreqBuftmp,src,tag=startTag))
                    jobreqBuf.append(jobreqBuftmp)
                    pos = int(result[3])
                    KH = int(result[1])
                    KW = int(result[2])
                    s = result[0]
                    if(s<scores[pos]):
                        print(f'Rank {rank} received Result from {src} : {result}, {s} < {scores[pos]} ({KH},{KW})')
                        updatecount+=1
                        scores[pos]=s
                        current_gen[pos]=[KH,KW]
                    else:
                        print(f'Rank {rank} received Result from {src}: {result}, {s} > {scores[pos]}')
                    joblist[pos]=0
            if(np.sum(joblist)==0):
                Jobdone=True
            counter+=1
        print(f'Rank {rank} DE Run {r:3d} End , joblist {np.sum(joblist)}')
        end = time.time()
        currentbest = np.min(scores)
        currentmean = np.mean(scores)
        sq75 = np.quantile(scores,0.75)
        sq25 = np.quantile(scores,0.25)
        smedian = np.median(scores)
        currentbestidx = np.argmin(scores)
        median = np.median(current_gen,axis=0)
        mean = np.mean(current_gen,axis=0)
        q75=np.quantile(current_gen,0.75,axis=0)
        q25=np.quantile(current_gen,0.25,axis=0)
        if(currentbest<overallBest):
            overallBest=currentbest
            overallBestConfig = current_gen[currentbestidx]
            bestGen = r
        t = (end-start)
        tb.add_scalars('KernerlSize H',{'q75':q75[0],'median':median[0],'mean':mean[0],'q25':q25[0]},r)
        tb.add_scalars('KernerlSize W',{'q75':q75[1],'median':median[1],'mean':mean[1],'q25':q25[1]},r)
        tb.add_scalars('Scores',{'q75':sq75,'median':smedian,'mean':currentmean,'q25':sq25},r)
        print(f'Rank {rank} DE Run {r:3d} CurrentBest: {currentbest:10.8f}, Mean: {currentmean:10.8f}, OverallBest: {overallBest:10.8f}/{bestGen:3d}, config: {current_gen[currentbestidx]}, updatecount: {updatecount:3d}, Generation RunTime: {t:10.8f}')
        print(f'Rank {rank} DE Run {r:3d} Kernel H q25: {q25[0]}, Mean: {mean[0]}, Median: {median[0]}, q75: {q75[0]}')
        print(f'Rank {rank} DE Run {r:3d} Kernel W q25: {q25[1]}, Mean: {mean[1]}, Median: {median[1]}, q75: {q75[1]}')
    print(f'Rank {rank} DE Run Completed : Best Score: {overallBest} , Config: {overallBestConfig}, find in Gen: {bestGen}')
    return

def runWorker(C,H,W,trainingset,validationset):
    count=0
    while(True):
        print(f'{rank} Ask for job')
        r = np.full(1,rank,dtype=int)
        comm.Send(r,0,tag=startTag)
        buf = np.zeros(3,dtype=int)
        s =MPI.Status()
        comm.Recv(buf,source=0,tag=MPI.ANY_TAG,status=s)
        if(s.tag==jobTag):
            #print(f'\tRank {rank} - Wait for job')
            print(f'{rank} received ksize: {buf}')
            pos = int(buf[2])
            KH = int(buf[0])
            KW = int(buf[1])
            r = fit((KH,KW),pos,rank,C,H,W,trainingset,validationset)
            r = np.array([r,KH,KW,pos,rank],dtype=float)
            print(f'{rank} job comp result: {r}')
            comm.Send(r,0,tag=jobTag)
            count+=1
        elif(s.tag==waitTag):
            print(f'{rank} no Job to exec, sleep for {sleep}s')
            time.sleep(sleep)

C = training.shape[1]
H = training.shape[2]
W = training.shape[3]


if(rank==0):
    H=np.random.choice(range(kernelMinH,kernelMaxH+1),pplSize,replace=True)
    W=np.random.choice(range(kernelMinW,kernelMaxW+1),pplSize,replace=True)
    kernelSizeList = [ [H[i],W[i]] for i in range(pplSize)  ]
    run()
else:
    fresult = loadData()
    training = torch.tensor(fresult).float()
    length=[x for x in range(fresult.shape[0])]
    t=int(np.floor(fresult.shape[0]*0.7))
    np.random.shuffle(length)
    tidxs=length[:t]
    vidxs=length[t:]
    runWorker(C,H,W,tidxs,vidxs)


#sys.stdout=open(outputDir+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),"w")
