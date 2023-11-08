#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

amise_normal = np.array(pd.read_csv(r'simulation\synthetic\synthetic_amise_normal.csv',header=None))
average_normal = np.array(pd.read_csv(r'simulation\synthetic\synthetic_average_normal.csv',header=None))
exp_normal = np.array(pd.read_csv(r'simulation\synthetic\synthetic_exponential_normal.csv',header=None))
amise_over = np.array(pd.read_csv(r'simulation\synthetic\synthetic_amise_oversmooth.csv',header=None))
average_over = np.array(pd.read_csv(r'simulation\synthetic\synthetic_average_oversmooth.csv',header=None))
exp_over = np.array(pd.read_csv(r'simulation\synthetic\synthetic_exponential_oversmooth.csv',header=None))
amise_cv = np.array(pd.read_csv(r'simulation\synthetic\synthetic_amise_cv.csv',header=None))
average_cv = np.array(pd.read_csv(r'simulation\synthetic\synthetic_average_cv.csv',header=None))
exp_cv = np.array(pd.read_csv(r'simulation\synthetic\synthetic_exponential_cv.csv',header=None))


#%%log

algo = ["amise","average","exponential"]
normal = [amise_normal,average_normal,exp_normal]
oversmooth = [amise_over,average_over,exp_over]
band = [normal,oversmooth]
frame = np.arange(100,600,100)


fig,axs = plt.subplots(len(band),len(frame))

width = np.arange(0.1,5,0.2)

for i in range(len(band)):
    for j in range(len(frame)):
        for k in range(3):
            y = band[i][k][band[i][k][:,1]==frame[j],2]
            yerr = band[i][k][band[i][k][:,1]==frame[j],3]
            axs[i,j].plot(width,y,label=algo[k])
            axs[i,j].errorbar(width, y, yerr=yerr, fmt='o')

axs[-1,-1].legend(prop={'size': 15})
plt.show()


#%%log
algo = ["amise","average","exponential"]
normal = [amise_cv,average_cv,exp_cv]
cut = np.arange(0.9,5,1)

width = np.arange(0.1,1.5,0.1)

fig,axs = plt.subplots(1,len(cut))

for j in range(len(cut)):
    for k in range(3):
        y = normal[k][abs(normal[k][:,0]-cut[j])<0.001,2]
        yerr = normal[k][abs(normal[k][:,0]-cut[j])<0.001,3]
        axs[j].plot(width,y,label=algo[k])
        axs[j].errorbar(width, y, yerr=yerr, fmt='o')
axs[-1].legend(prop={'size': 16})

plt.show()
