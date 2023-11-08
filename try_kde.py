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

def multiprocess_TAKDE(data, w=20,mc = 10,re=True,cv=(32/3)**(1/5)):
    import numpy as np
    log = np.empty(mc)
    for i in range(mc):
        np.random.seed()
        x = np.arange(0.05,2,0.1)
        df_1, df_1t = TK.split(data,0.1)
        df_1_burnin, df_1_test = TK.burnin(df_1,1)
        dum1, df_1t = TK.burnin(df_1t,1)
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
        file1 = pd.read_csv('/scratch/wang.yinso/cps2021/D1.txt',sep = '  ',header = None)
        Data_1 = np.array(file1)
        temp1 = np.sqrt(Data_1[:,1]*Data_1[:,2])
        D1 = np.transpose(np.vstack((Data_1[:,0],temp1)))
        data1 = TK.mean_size(D1,1)
        len_list = np.arange(8,60,4)
        cv_list = np.arange(0.05,1.1,0.05)
        cutoff_list = list(product(len_list,cv_list))
        pcount = int(len(cutoff_list)/(size-1))
        mod = len(cutoff_list)-(size-1)*pcount
        
        if size > len(cutoff_list):
            for p in range(1,len(cutoff_list)+1):
                comm.send([cutoff_list[p-1],],dest=p,tag=1)
                comm.send(data1,dest=p,tag=2)
            
            globallist = []
            for p in range(1,len(cutoff_list)+1):
                data = comm.recv(source=p,tag=3)
                globallist.extend(data)
            cv_result = np.vstack(globallist)
            savetxt('/scratch/wang.yinso/cps2021/simulation/tem/cvresult_KDE.csv',cv_result,delimiter=',')
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
                
            globallist = []
            for p in range(1,size):
                data = comm.recv(source=p,tag=3)
                globallist.extend(data)
            cv_result = np.vstack(globallist)
            
            savetxt('/scratch/wang.yinso/cps2021/simulation/tem/cvresult_KDE.csv',cv_result,delimiter=',')
    
    else:
        print("Node %d Received" %rank)
        locallist = comm.recv(source=0,tag=1)
        localdata = comm.recv(source=0,tag=2)
        localresult = []
        for i in range(len(locallist)):
            result = np.empty(4)
            w = locallist[i][0]
            cv = locallist[i][1]
            log,ste = multiprocess_TAKDE(localdata,w=w,cv=cv)
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
