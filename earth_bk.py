import sys
sys.path.append('/scratch/wang.yinso/cps2021')
import ReTAKDE as TK
import BK as BK
import numpy as np
from numpy import savetxt
import pandas as pd
from mpi4py import MPI
import time

def multiprocess_TAKDE(data,testlist, n=20,noise = 0.1, mc=10):
    import numpy as np
    log = np.empty(mc)
    for k in range(mc):
        x = np.arange(0.05,2,0.1)
        df_1_test, df_1t = TK.splitvlist(data,testlist)
        estimator = BK.BK(x,n=n)
        Y1 = estimator.create_bin(df_1_test)
        mat_cov = noise*np.identity(n)
        for i in range(n):
            if i < 2:
                mat_cov[i,i] = noise
            else:
                mat_cov[i,i] = noise/16
        a0 = np.ones(n)
        w0 = 0.01*np.identity(n)
        B = estimator.Bspline_basis()
        params = estimator.Reg_BK(a0,w0,Y1,B,mat_cov)
        ests=[]
        for i in range(len(params)):
            ests.append(estimator.multi_eval(df_1t[i],params[i],reg=True))
        testlog = 0
        tot = 0
        for j in range(len(ests)):
            tot += len(ests[j])
            testlog += sum(np.log(ests[j]))
        if np.isnan(testlog):
            testlog = -10000
        else:
            testlog = max(testlog/tot,-10000)
        log[k] = testlog
        
    return np.mean(log), np.std(log)/np.sqrt(mc)

def main():
    StartTime = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank ==0:
        tsv_data = pd.read_csv("/scratch/wang.yinso/cps2021/Earthquakes_TRAIN.tsv",sep='\t')
        tsv_data = np.array(tsv_data)
        D = []
        for i in range(1,len(tsv_data[0])):
            D.append(tsv_data[:,i])
        data1 = TK.standard(D,upper=2,lower=0,ratio=1)
        testlist = np.random.randint(5,20,len(data1))
        cutoff_list = np.arange(0.01,1,0.01)
        pcount = int(len(cutoff_list)/(size-1))
        mod = len(cutoff_list)-(size-1)*pcount
        
        if size > (len(cutoff_list)):
            for p in range(1,len(cutoff_list)+1):
                comm.send([cutoff_list[p-1],],dest=p,tag=4)
                comm.send(data1,dest=p,tag=1)
                comm.send(testlist,dest=p,tag=2)
            
            globallist = []
            for p in range(1,len(cutoff_list)+1):
                data = comm.recv(source=p,tag=3)
                globallist.extend(data)
            cv_result = np.vstack(globallist)
            savetxt('/scratch/wang.yinso/cps2021/simulation/earth/cvresult_BK.csv',cv_result,delimiter=',')
        else:
            for p in range(1,size):
                if (p-1)<mod:
                    Start = (pcount+1)*(p-1)
                    End = min((pcount+1)*p,len(cutoff_list))
                else:
                    Start = (pcount+1)*mod+pcount*(p-1-mod)
                    End = min(((pcount+1)*mod+pcount*(p-mod)),len(cutoff_list))
                comm.send(cutoff_list[Start:End],dest=p,tag=4)
                comm.send(data1,dest=p,tag=1)
                comm.send(testlist,dest=p,tag=2)
            
            globallist = []
            for p in range(1,size):
                data = comm.recv(source=p,tag=3)
                globallist.extend(data)
            cv_result = np.vstack(globallist)
            savetxt('/scratch/wang.yinso/cps2021/simulation/earth/cvresult_BK.csv',cv_result,delimiter=',')
    
    else:
        print("Node %d Received" %rank)
        localdata = comm.recv(source=0,tag=1)
        locallist = comm.recv(source=0,tag=2)
        no = comm.recv(source=0,tag=4)
        localresult = []
        for i in range(len(no)):
            result = np.empty(3)
            noise = no[i]
            log,ste = multiprocess_TAKDE(localdata,locallist,noise=noise)
            result[0] = noise
            result[1] = log
            result[2] = ste
            localresult.append(result)
        
        comm.send(localresult,dest=0,tag=3)
        print("Node %d job finished" %rank)
    print("total elapsed" + str(time.time()-StartTime))


if __name__ == "__main__":
    main()