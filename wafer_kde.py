import sys
sys.path.append('/scratch/wang.yinso/cps2021')
import ReTAKDE as TK
import KDE
import numpy as np
from numpy import savetxt
import pandas as pd
from mpi4py import MPI
import time
from itertools import product

def multiprocess_TAKDE(data, testlist, w=20,mc = 100,re=False,cv=0):
    import numpy as np
    log = np.empty(mc)
    for i in range(mc):
        x = np.arange(0.05,2,0.1)
        df_1_test, df_1t = TK.splitvlist(data,testlist)
        estimator = KDE.KDEtrack(x,w=w,cv=cv)
        ests = estimator.Streaming(df_1t[(w-1):], df_1_test,re=re)
        testlog = 0
        tot = 0
        for j in range(len(ests)):
            testlog += sum(np.log(ests[j]))
            tot += len(ests[j])
        if np.isnan(testlog):
            testlog = -10000
        else:
            testlog = max(testlog/tot,-10000)
        log[i] = testlog
    return np.mean(log),np.std(log)/np.sqrt(mc)

def main():
    StartTime = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank ==0:
        tsv_data = pd.read_csv("/scratch/wang.yinso/cps2021/Wafer_TRAIN.tsv",sep='\t')
        tsv_data = np.array(tsv_data)
        D = []
        for i in range(1,len(tsv_data[0])):
            D.append(tsv_data[(tsv_data[:,0]==1),i])
        data1 = TK.standard(D,upper=2,lower=0,ratio=1)
        testlist = np.random.randint(5,20,len(data1))
        len_list = np.arange(20,35,1)
        cv_list = np.arange(0.05,1.1,0.05)
        cutoff_list = list(product(len_list,cv_list))
        pcount = int(len(cutoff_list)/(size-1))
        mod = len(cutoff_list)-(size-1)*pcount
        
        if size > len(cutoff_list):
            for p in range(1,len(cutoff_list)+1):
                comm.send([cutoff_list[p-1],],dest=p,tag=1)
                comm.send(data1,dest=p,tag=2)
                comm.send(testlist,dest=p,tag=4)
            
            globallist = []
            for p in range(1,len(cutoff_list)+1):
                data = comm.recv(source=p,tag=3)
                globallist.extend(data)
            cv_result = np.vstack(globallist)
            savetxt('/scratch/wang.yinso/cps2021/simulation/wafer/cvresult_kde.csv',cv_result,delimiter=',')
        else:
            for p in range(1,size):
                if (p-1)<mod:
                    Start = (pcount+1)*(p-1)
                    End = min((pcount+1)*p,len(cutoff_list))
                else:
                    Start = (pcount+1)*mod+pcount*(p-1-mod)
                    End = min(((pcount+1)*mod+pcount*(p-mod)),len(cutoff_list))
                comm.send(cutoff_list[Start:End],dest=p,tag=1)
                comm.send(data1,dest=p,tag=2)
                comm.send(testlist,dest=p,tag=4)
            
            globallist = []
            for p in range(1,size):
                data = comm.recv(source=p,tag=3)
                globallist.extend(data)
            cv_result = np.vstack(globallist)
            savetxt('/scratch/wang.yinso/cps2021/simulation/wafer/cvresult_kde.csv',cv_result,delimiter=',')
    
    else:
        print("Node %d Received" %rank)
        locallist = comm.recv(source=0,tag=1)
        localdata = comm.recv(source=0,tag=2)
        testlist = comm.recv(source=0,tag=4)
        localresult = []
        for i in range(len(locallist)):
            result = np.empty(4)
            w = locallist[i][0]
            cv = locallist[i][1]
            log,ste = multiprocess_TAKDE(localdata,testlist = testlist, w=w,cv=cv)
            result[0] = w
            result[1] = cv
            result[2] = log
            result[3] = ste
            localresult.append(result)
        comm.send(localresult,dest=0,tag=3)
        print("Node %d job finished" %rank)
    print("total elapsed" + str(time.time()-StartTime))


if __name__ == "__main__":
    main()
