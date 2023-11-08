#%% import packages
import numpy as np
import matplotlib.pyplot as plt

#%% Generate Density Vectors


x = np.linspace(-4,4,2000)
densitylist = []
gen = GaussianMixture(n=100)


for i in range(15):
    densitylist.append(gen.GMM_pdf(x,ind=i))
    
#%%

row,col = 3,5

fig,ax = plt.subplots(3,5,sharex=True)
names = list(gen.densitydict.keys())

for i in range(15):
    rowind = int(i/col)
    colind = i%col
    ax[rowind,colind].plot(x,densitylist[i],lw=0.5)
    ax[rowind,colind].set_title(names[i],fontsize=8,fontname="Times New Roman")
    ax[rowind,colind].set_xticks([])
    ax[rowind,colind].set_yticks([])

fig.tight_layout()
plt.savefig('GMM_plots.pdf')

