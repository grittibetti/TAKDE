import sys
sys.path.append('/scratch/wang.yinso/cps2021')
import ReTAKDE as TK
import numpy as np
from numpy import savetxt
from itertools import product
from mpi4py import MPI
import time
import DataGenerator as dg

def multiprocess_TAKDE(frame=100,cutoff = 4, mc = 100, rule = "normal", weighting = "average"):
    import numpy as np
    log = np.empty(mc)
    n=500
    gene = dg.GaussianMixture(frame)
    for i in range(mc):
        testlist = np.empty(frame)
        for j in range(len(testlist)):
            testlist[j] = np.random.randint(5,50)
        testlist = testlist.astype(int)
        lenlist = testlist+n
        data = gene.Generation(lenlist)
        df_1_test, df_1t = TK.splitvlist(data,testlist)
        estimator = TK.TAKDE(cutoff=cutoff)
        ests = estimator.Streaming_Estimation(df_1_test,df_1t,width_selector = rule, weighting = weighting)
        testlog = 0
        for j in range(len(ests)):
            testlog += sum(np.log(ests[j]))/(frame*n)
        if np.isnan(testlog):
            testlog = -10000
        log[i] = max(testlog,-10000)
    return np.mean(log),np.std(log)/np.sqrt(mc)

def main():
    StartTime = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank ==0:
        cutoff_list = np.arange(0.1,5,0.2)
        frame_list = np.arange(100,600,100)
        iter_list = list(product(cutoff_list,frame_list))
        pcount = int(len(iter_list)/(size-1))
        mod = len(iter_list)-(size-1)*pcount
        
        if size > len(iter_list):
            for p in range(1,len(iter_list)+1):
                comm.send([iter_list[p-1],],dest=p,tag=1)
            
            globallist = []
            for p in range(1,len(iter_list)+1):
                data = comm.recv(source=p,tag=2)
                globallist.extend(data)
            cv_result = np.vstack(globallist)
            savetxt('/scratch/wang.yinso/cps2021/simulation/synthetic/synthetic_average_normal.csv',cv_result,delimiter=',')
        else:
            for p in range(1,size):
                if (p-1)<mod:
                    Start = (pcount+1)*(p-1)
                    End = min((pcount+1)*p,len(iter_list))
                else:
                    Start = (pcount+1)*mod+pcount*(p-1-mod)
                    End = min(((pcount+1)*mod+pcount*(p-mod)),len(iter_list))
                comm.send(iter_list[Start:End],dest=p,tag=1)
            
            globallist = []
            for p in range(1,size):
                data = comm.recv(source=p,tag=2)
                globallist.extend(data)
            cv_result = np.vstack(globallist)
            savetxt('/scratch/wang.yinso/cps2021//simulation/synthetic/synthetic_average_normal.csv',cv_result,delimiter=',')
    
    else:
        print("Node %d Received" %rank)
        locallist = comm.recv(source=0,tag=1)
        localresult = []
        for i in range(len(locallist)):
            result = np.empty(4)
            cutoff = locallist[i][0]
            frame = locallist[i][1]
            log,ste = multiprocess_TAKDE(frame=frame,cutoff=cutoff)
            result[0] = cutoff
            result[1] = frame
            result[2],result[3] = log,ste
            localresult.append(result)
        comm.send(localresult,dest=0,tag=2)
        print("Node %d job finished" %rank)
    print("total elapsed" + str(time.time()-StartTime))


if __name__ == "__main__":
    main()