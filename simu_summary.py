#%% Get results summary for earth dataset
import numpy as np
import pandas as pd

amise_cv_earth = np.array(pd.read_csv(r'simulation\earth\cvresult_amise_cv.csv', header=None))
amise_normal_earth = np.array(pd.read_csv(r'simulation\earth\cvresult_amise_normal.csv', header=None))
BK_earth = np.array(pd.read_csv(r'simulation\earth\cvresult_BK.csv', header=None))
kde_earth = np.array(pd.read_csv(r'simulation\earth\cvresult_kde.csv', header=None))
kernel_earth = np.array(pd.read_csv(r'simulation\earth\cvresult_k.csv', header=None))

cv_max = np.argmax(amise_cv_earth[:, 2])
best_amise_cv, best_amise_cv_std = amise_cv_earth[cv_max, 2], amise_cv_earth[cv_max, 3]
normal_max = np.argmax(amise_normal_earth[:, 1])
best_amise_normal, best_amise_normal_std = amise_normal_earth[normal_max, 1], amise_normal_earth[normal_max, 2]
BK_max = np.argmax(BK_earth[:, 1])
best_BK, best_BK_std = BK_earth[BK_max, 1], BK_earth[BK_max, 2]
kde_max = np.argmax(kde_earth[:, 2])
best_kde, best_kde_std = kde_earth[kde_max, 2], kde_earth[kde_max, 3]
kernel_max = np.argmax(kernel_earth[:, 1])
best_kernel, best_kernel_std = kernel_earth[kernel_max, 1], kernel_earth[kernel_max, 2]

# %%Get results summary for ECG dataset
amise_cv_ecg = np.array(pd.read_csv(r'simulation\ECG\cvresult_amise_cv.csv', header=None))
amise_normal_ecg = np.array(pd.read_csv(r'simulation\ECG\cvresult_amise_normal.csv', header=None))
BK_ecg = np.array(pd.read_csv(r'simulation\ECG\cvresult_BK.csv', header=None))
kde_ecg = np.array(pd.read_csv(r'simulation\ECG\cvresult_kde.csv', header=None))
kernel_ecg = np.array(pd.read_csv(r'simulation\ECG\cvresult_k.csv', header=None))

cv_max = np.argmax(amise_cv_ecg[:, 2])
best_amise_cv, best_amise_cv_std = amise_cv_ecg[cv_max, 2], amise_cv_ecg[cv_max, 3]
normal_max = np.argmax(amise_normal_ecg[:, 1])
best_amise_normal, best_amise_normal_std = amise_normal_ecg[normal_max, 1], amise_normal_ecg[normal_max, 2]
BK_max = np.argmax(BK_ecg[:, 1])
best_BK, best_BK_std = BK_ecg[BK_max, 1], BK_ecg[BK_max, 2]
kde_max = np.argmax(kde_ecg[:, 2])
best_kde, best_kde_std = kde_ecg[kde_max, 2], kde_ecg[kde_max, 3]
kernel_max = np.argmax(kernel_ecg[:, 1])
best_kernel, best_kernel_std = kernel_ecg[kernel_max, 1], kernel_ecg[kernel_max, 2]

# %%Get results summary for star dataset
amise_cv_star = np.array(pd.read_csv(r'simulation\star\cvresult_amise_cv.csv', header=None))
amise_normal_star = np.array(pd.read_csv(r'simulation\star\cvresult_amise_normal.csv', header=None))
BK_star = np.array(pd.read_csv(r'simulation\star\cvresult_BK.csv', header=None))
kde_star = np.array(pd.read_csv(r'simulation\star\cvresult_kde.csv', header=None))
kernel_star = np.array(pd.read_csv(r'simulation\star\cvresult_k.csv', header=None))

cv_max = np.argmax(amise_cv_star[:, 2])
best_amise_cv, best_amise_cv_std = amise_cv_star[cv_max, 2], amise_cv_star[cv_max, 3]
normal_max = np.argmax(amise_normal_star[:, 1])
best_amise_normal, best_amise_normal_std = amise_normal_star[normal_max, 1], amise_normal_star[normal_max, 2]
BK_max = np.argmax(BK_star[:, 1])
best_BK, best_BK_std = BK_star[BK_max, 1], BK_star[BK_max, 2]
kde_max = np.argmax(kde_star[:, 2])
best_kde, best_kde_std = kde_star[kde_max, 2], kde_star[kde_max, 3]
kernel_max = np.argmax(kernel_star[:, 1])
best_kernel, best_kernel_std = kernel_star[kernel_max, 1], kernel_star[kernel_max, 2]

# %%Get results summary for wafer dataset
amise_cv_wafer = np.array(pd.read_csv(r'simulation\wafer\cvresult_amise_cv.csv', header=None))
amise_normal_wafer = np.array(pd.read_csv(r'simulation\wafer\cvresult_amise_normal.csv', header=None))
BK_wafer = np.array(pd.read_csv(r'simulation\wafer\cvresult_BK.csv', header=None))
kde_wafer = np.array(pd.read_csv(r'simulation\wafer\cvresult_kde.csv', header=None))
kernel_wafer = np.array(pd.read_csv(r'simulation\wafer\cvresult_k.csv', header=None))

cv_max = np.argmax(amise_cv_wafer[:, 2])
best_amise_cv, best_amise_cv_std = amise_cv_wafer[cv_max, 2], amise_cv_wafer[cv_max, 3]
normal_max = np.argmax(amise_normal_wafer[:, 1])
best_amise_normal, best_amise_normal_std = amise_normal_wafer[normal_max, 1], amise_normal_wafer[normal_max, 2]
BK_max = np.argmax(BK_wafer[:, 1])
best_BK, best_BK_std = BK_wafer[BK_max, 1], BK_wafer[BK_max, 2]
kde_max = np.argmax(kde_wafer[:, 2])
best_kde, best_kde_std = kde_wafer[kde_max, 2], kde_wafer[kde_max, 3]
kernel_max = np.argmax(kernel_wafer[:, 1])
best_kernel, best_kernel_std = kernel_wafer[kernel_max, 1], kernel_wafer[kernel_max, 2]
# %%Get results summary for tem dataset
amise_cv_tem = np.array(pd.read_csv(r'simulation\tem\cvresult_amise_cv.csv', header=None))
amise_normal_tem = np.array(pd.read_csv(r'simulation\tem\cvresult_amise_normal.csv', header=None))
BK_tem = np.array(pd.read_csv(r'simulation\tem\cvresult_BK.csv', header=None))
kde_tem = np.array(pd.read_csv(r'simulation\tem\cvresult_KDE.csv', header=None))
kernel_tem = np.array(pd.read_csv(r'simulation\tem\cvresult_K.csv', header=None))

cv_max = np.argmax(amise_cv_tem[:, 2])
#This result need to be divided by 1335, which is the total number of test data
best_amise_cv, best_amise_cv_std = amise_cv_tem[cv_max, 2]/1335, amise_cv_tem[cv_max, 3]/1335
normal_max = np.argmax(amise_normal_tem[:, 1])
best_amise_normal, best_amise_normal_std = amise_normal_tem[normal_max, 1], amise_normal_tem[normal_max, 2]
BK_max = np.argmax(BK_tem[:, 1])
best_BK, best_BK_std = BK_tem[BK_max, 1], BK_tem[BK_max, 2]
kde_max = np.argmax(kde_tem[:, 2])
best_kde, best_kde_std = kde_tem[kde_max, 2], kde_tem[kde_max, 3]
kernel_max = np.argmax(kernel_tem[:, 1])
best_kernel, best_kernel_std = kernel_tem[kernel_max, 1], kernel_tem[kernel_max, 2]
# %%
