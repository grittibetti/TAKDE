#%%
import numpy as np
import time
import pandas as pd
import ReTAKDE as TK
from itertools import product
from numpy import savetxt


#%%

gen = GaussianMixture(6000)

samplelst = np.random.randint(5,20,6000)

data = gen.Generation(samplelst)

#%% Tem
file1 = pd.read_csv('E:\Texas_AM_University\Research\CPS\python\Kalman_Filter\D1.txt',sep = '  ',header = None)
Data_1 = np.array(file1)
temp1 = np.sqrt(Data_1[:,1]*Data_1[:,2])
D1 = np.transpose(np.vstack((Data_1[:,0],temp1)))
data = mean_size(D1,1)

#%% ECG

tsv_data = pd.read_csv("E:\Texas_AM_University\Research\CPS\python\Kalman_Filter\ECGtest.tsv",sep='\t')
tsv_data = np.array(tsv_data)
D = []
for i in range(1,len(tsv_data[0])):
    D.append(tsv_data[(tsv_data[:,0]==3),i])
data = standard(D,upper=2,lower=0,ratio=0.8)

#%%Wafer

tsv_data = pd.read_csv("E:\Texas_AM_University\Research\CPS\python\Kalman_Filter\Wafer_TRAIN.tsv",sep='\t')
tsv_data = np.array(tsv_data)
D = []
for i in range(1,len(tsv_data[0])):
    D.append(tsv_data[(tsv_data[:,0]==1),i])
data = standard(D,upper=2,lower=0,ratio=0.8)

#%% Earth
tsv_data = pd.read_csv("E:\Texas_AM_University\Research\CPS\python\Kalman_Filter\Earthquakes_TRAIN.tsv",sep='\t')
tsv_data = np.array(tsv_data)
D = []
for i in range(1,len(tsv_data[0])):
    D.append(tsv_data[:,i])
data = standard(D,upper=2,lower=0,ratio=1)

#%% Star

tsv_data = pd.read_csv("E:\Texas_AM_University\Research\CPS\python\Kalman_Filter\StarLightCurves_TRAIN.tsv",sep='\t')
tsv_data = np.array(tsv_data)
D = []
for i in range(1,len(tsv_data[0])):
    D.append(tsv_data[:,i])
data = standard(D,upper=2,lower=0,ratio=1)

#%%
test = []

for i in range(len(data)):
    
    test.append([1]*1)


#%%

TK = TAKDE(cutoff = 0.2)

start = time.perf_counter()

trail = TK.Streaming_Estimation(data, test, width_list = 0, 
                             width_selector="normal", weighting = "amise",cv = 0,
                             fixedwindow=False,t=10)

end = time.perf_counter()

print(end-start)

#%%

x = np.linspace(0.05,1.95,20)
mat_cov = np.identity(20)
for i in range(20):
    if i < 2:
        mat_cov[i,i] = 0.064
    else:
        mat_cov[i,i] = 0.00382

a0 = np.ones(20)
w0 = 0.01*np.identity(20)

Bsp = BK(x,n=20)

datamass = Bsp.create_bin(data)

B = Bsp.Bspline_basis()

start = time.perf_counter()

params = Bsp.Reg_BK(a0,w0,datamass,B,mat_cov,epsilon = 0.0001,t = 10)

end = time.perf_counter()

print(end-start)

start = time.perf_counter()
Bsp.point_eval(1,params[10],reg=True)
Bsp.point_eval(1,params[11],reg=True)
Bsp.point_eval(1,params[12],reg=True)
Bsp.point_eval(1,params[13],reg=True)
end = time.perf_counter()

print(end-start)


#%%
w = 20
Ktrack = KDEtrack(x,w)

start = time.perf_counter()
trail = Ktrack.Streaming(test[w-1:],data,re=True)
end = time.perf_counter()

print(end-start)



