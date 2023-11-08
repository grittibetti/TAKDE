import sys
sys.path.append('/scratch/wang.yinso/cps2021')
import ReTAKDE as TK
import numpy as np
from numpy import savetxt
import pandas as pd
from mpi4py import MPI
import time

def stream_kde(obs,test,cv=0):
    
    from sklearn.neighbors import KernelDensity
    
    if not len(obs)==len(test):
        raise ValueError("test list length does not equal to train list")
        
    else:
        tot = 0
        log = 0
        for i in range(len(obs)):
            tot += len(obs[i])
            sigma = cv*np.std(obs[i])*len(obs[i])**(-1/5)
            x = obs[i].reshape(-1,1)
            kde = KernelDensity(kernel='gaussian',bandwidth=sigma).fit(x)
            if len(test[i]) == 0:
                log+=0
            else:
                y = test[i].reshape(-1,1)
                log += sum(kde.score_samples(y))
        
        if np.isnan(log):
            log = -10000
        else:
            log/=tot
        return log


def multiprocess_TAKDE(data, mc = 100,cv=0):
    import numpy as np
    log = np.empty(mc)
    for i in range(mc):
        df_1, df_1t = TK.split(data,0.1)
        df_1_burnin, df_1_test = TK.burnin(df_1,1)
        dum1, df_1t = TK.burnin(df_1t,1)
        testlog = stream_kde(df_1_test,df_1t,cv=cv)
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
        cutoff_list = np.arange(0.1,2,0.02)
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
            savetxt('/scratch/wang.yinso/cps2021/simulation/tem/cvresult_K.csv',cv_result,delimiter=',')
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
            
            savetxt('/scratch/wang.yinso/cps2021/simulation/tem/cvresult_K.csv',cv_result,delimiter=',')
    
    else:
        print("Node %d Received" %rank)
        locallist = comm.recv(source=0,tag=1)
        localdata = comm.recv(source=0,tag=2)
        localresult = []
        for i in range(len(locallist)):
            result = np.empty(3)
            cv = locallist[i]
            log,ste = multiprocess_TAKDE(localdata,cv=cv)
            result[0] = cv
            result[1] = log
            result[2] = ste
            localresult.append(result)
        comm.send(localresult,dest=0,tag=3)
        print("Node %d job finished" %rank)
    print("total elapsed" + str(time.time()-StartTime))


if __name__ == "__main__":
    main()