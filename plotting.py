#%% import packages
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz
import matplotlib.animation as ani
import scipy.stats as stat
from scipy.integrate import quad

#%%
lower = 0
upper = 2
inter = 0.1
n = int((upper-lower)/inter)
index = np.arange((lower+inter/2),(upper+inter/2),inter)

#%% read data
file1 = pd.read_csv('D1.txt',sep = '  ',header = None)
Data_1 = np.array(file1)
temp1 = np.sqrt(Data_1[:,1]*Data_1[:,2])
D1 = np.transpose(np.vstack((Data_1[:,0],temp1)))
data1 = mean_size(D1,1)

#%% Wafer

tsv_data = pd.read_csv("Wafer_TRAIN.tsv",sep='\t')
tsv_data = np.array(tsv_data)
D = []
for i in range(1,len(tsv_data[0])):
    D.append(tsv_data[(tsv_data[:,0]==1),i])
data1 = standard(D,upper=2,lower=0,ratio=0.8)

#%% ECG

tsv_data = pd.read_csv("ECGtest.tsv",sep='\t')
tsv_data = np.array(tsv_data)
D = []
for i in range(1,len(tsv_data[0])):
    D.append(tsv_data[(tsv_data[:,0]==3),i])
data1 = standard(D,upper=2,lower=0,ratio=0.8)


#%%
itr = np.arange(0,2,0.005)
test_list = [itr]*len(data1)

#%% estimates TAKDE
estor = TAKDE(cutoff=1)
# =============================================================================
# amise = estor.Streaming_Estimation(data1,test_list,width_selector = 'normal', weighting = 'amise')
# amise = estor.Streaming_Estimation(data1,test_list,width_selector = 'normal', weighting = 'amise')
# average = estor.Streaming_Estimation(data1,test_list,width_selector = 'normal', weighting = 'average')
# =============================================================================
cv = estor.Streaming_Estimation(data1,test_list,width_selector = 'cv', weighting = 'amise',cv = 0.45)
cv2 = estor.Streaming_Estimation(data1,test_list,width_selector = 'cv', weighting = 'amise',cv = (32/3)**(1/5))

#%% BK

x = np.linspace(0.05,1.95,20)
mat_cov = np.identity(20)
for i in range(20):
    if i < 2:
        mat_cov[i,i] = 0.82
    else:
        mat_cov[i,i] = 0.05

a0 = np.ones(20)
w0 = 0.01*np.identity(20)

Bsp = BK(x,n=20)

datamass = Bsp.create_bin(data1)

B = Bsp.Bspline_basis()

params = Bsp.Reg_BK(a0,w0,datamass,B,mat_cov,epsilon = 0.001,t = 10)

Best = []

for i in range(len(params)):
    Best.append(Bsp.multi_eval(itr,params[i],reg=True))
    
#%% KDEtrack
x = np.linspace(0,2,20)
w = 30
Ktrack = KDEtrack(x,w,cv=0.45)
test = [itr]*len(data1)
trail = Ktrack.Streaming(test[w-1:],data1,re=False)


#%% plots
time = 225

mass = (1/len(data1[time]))*n*np.ones(len(data1[time]))

#plt.hist(data1[time],weights = mass,bins = index)
plt.hist(data1[time])
# =============================================================================
# plt.plot(itr,amise[time])
# plt.plot(itr,exp[time])
# plt.plot(itr,average[time])
# =============================================================================

plt.plot(itr,cv[time],label="TAKDE(optimal)")
plt.plot(itr,cv2[time],label="TAKDE(normal)")
plt.plot(itr,Best[time],label="B-spline Kalman Filter")
plt.plot(itr,trail[time-w+1],label="KDEtrack")
plt.legend()

#%% 

time = [225,450,675,900]

fig,axs = plt.subplots(3,len(time),figsize = (40,5))

for j in range(len(time)):
    axs[0,j].hist(data1[time[j]])
    axs[1,j].hist(data1[time[j]])
    axs[2,j].hist(data1[time[j]])
    axs[0,j].plot(itr,cv[time[j]],label="TAKDE(optimal)")
    axs[0,j].plot(itr,cv2[time[j]],label="TAKDE(normal)")
    axs[1,j].plot(itr,Best[time[j]],label="B-spline Kalman Filter")
    axs[2,j].plot(itr,trail[time[j]-w+1],label="KDEtrack")
axs[0,0].legend(loc = 'upper right',fontsize = 'large')
axs[1,0].legend(loc = 'upper right',fontsize = 'large')
axs[2,0].legend(loc = 'upper right',fontsize = 'large')

plt.show()

#%%
tot_amise = 0
for i in range(len(amise)):
    tot_amise += sum(np.log(amise[i]))

see = []
for i in range(len(amise)):
    see.append(np.log(amise[i]))
    
tot_exp = 0
for i in range(len(exp)):
    tot_exp += sum(np.log(exp[i]))
    
tot_average = 0
for i in range(len(average)):
    tot_average += sum(np.log(average[i]))
    
#%%

