# TAKDE
The code repository for Temporal-Adaptive Kernel Density Estimator

#Package dependency:

numpy
scipy

#Data:

You would want a training data D_t \in \R^{T,n_t} as a list of numpy array for T time steps, n_t is the number of data at each time step, which could vary over time. You would also want evaluation D_test for evaluation, with the same time steps T as training data (pad it if you don't need that many time steps).

#Usage:

import ReTAKDE as TK

estimator = TK.TAKDE(cutoff=cutoff)

ests = estimator.Streaming_Estimation(D_test,D_t,width_selector = rule, weighting = weighting,cv=cv)

#Parameter Description:

cutoff:
The cutoff value, which decides the number of windows to keep in the memory, smaller cutoff value will results in faster TAKDE.

width_selector:
Can be "normal", which refers to normal bandwidth sequence, [M. P. Wand and M. C. Jones, Kernel Smoothing. CRC press, 1994]

"smooth", which refers to oversmoothed bandwidth sequence, [G. R. Terrell, “The maximal smoothing principle in density estimation,” Journal of the American Statistical Association, vol. 85, no. 410, pp. 470–477, 1990.]

"cv", custom smoothness parameter, which will activate the later cv variable with custom input value.

weighting:
Can be "average", which will be the naive average weighting KDE

"exponential", which will be an exponential decaying weighting for different time steps

"amise", the optimal weighting sequence for TAKDE.

cv:
only activated when width_selector == "cv", controls the smoothness parameter of the bandwidth sequence.
