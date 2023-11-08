# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:08:45 2021

@author: ronny
"""

#%% 2d plot for 
t = 15
num_mc = 100
density_est = np.zeros(len(df_1_test))
for mc in range(num_mc):
    df_1, df_1t = split(data_1,0.1)
    df_1_burnin, df_1_test = burnin(df_1,1)
    dum1, df_1t = burnin(df_1t,1)
    estimator = TAKDE(init_data = df_1_test[0],width = 0.02,alpha = 0.85,metric = "SecondOrderIntegration")
    for frame in range(len(df_1_test)):
        density_est[frame] += sum(estimator.Estimation(df_1_test[max(0,int(frame-t+1)):(frame+1)],df_1t[frame]))/(num_mc*max(1,len(df_1t[frame])))
        
#%% 2 frames

t = 2
num_mc = 100
density_est_2 = np.zeros(len(df_1_test))
for mc in range(num_mc):
    df_1, df_1t = split(data_1,0.1)
    df_1_burnin, df_1_test = burnin(df_1,1)
    dum1, df_1t = burnin(df_1t,1)
    estimator = TAKDE(init_data = df_1_test[0],width = 0.038,alpha = 0.7,metric = "SecondOrderIntegration")
    for frame in range(len(df_1_test)):
        density_est_2[frame] += sum(estimator.Estimation(df_1_test[max(0,int(frame-t+1)):(frame+1)],df_1t[frame]))/(num_mc*max(1,len(df_1t[frame])))
        
#%% plot 
ta = 600
tb = 800
plt.plot(density_est[ta:tb])
plt.plot(density_est_2[ta:tb])

#%% 
ta = 400
tb = 600
max_hist = np.zeros(1148)

for i in range(1148):
    max_hist[i] = max(hist[:,i])
    
plt.plot(density_est[ta:tb]/8+0.2)
plt.plot(density_est_2[ta:tb]/8+0.2)
plt.plot(max_hist[ta:tb])