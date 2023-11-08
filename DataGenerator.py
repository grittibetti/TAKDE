# -*- coding: utf-8 -*-
"""
Created on Sat Aug 3 19:55:21 2021

@author: ronny
"""
import numpy as np

class GaussianMixture():
    
    def __init__(self, n, m=15, upper = 2, lower = 0, numbin=20, shrink=1, idx = None):
        
        """
        The density dictionary is arranged in the order of name, weight, mean, and standard deviation. For each key, the value[0] is the weightlist, value[1] is the meanlist, value[2] is the stdlist.
        """
        
        
        self.nframe = n
        self.ndensity = m
        self.densitydict = {
                "Gaussian":([1,],[0,],[1,]),
                "Skewed Unimodel":([1/5,1/5,3/5],[0,1/2,13/12],[1,(2/3)**2,(5/9)**2]),
                "Strongly Skewed":(list(np.repeat(1/8,8)), list(3*((2/3)**np.arange(8)-1)), list((2/3)**(2*np.arange(8)))),
                "Kurtotic unimodel":([2/3,1/3],[0,0],[1,1/100]),
                "Outlier":([1/10,9/10],[0,0],[1,1/100]),
                "Bimodal":([1/2,1/2],[-1,1],[4/9,4/9]),
                "Separated bimodal":([1/2,1/2],[-3/2,3/2],[1/4,1/4]),
                "Skewed bimodal":([3/4,1/4],[0,3/2],[1,1/9]),
                "Trimodal":([9/20,9/20,1/10],[-6/5,6/5,0],[9/25,9/25,1/16]),
                "Claw":([1/2,1/10,1/10,1/10,1/10,1/10],[0,-1,-1/2,0,1/2,1],[1,1/100,1/100,1/100,1/100,1/100]),
                "Doube claw":(list(np.concatenate((np.repeat(49/100,2),np.repeat(1/350,7)))),list(np.concatenate((np.array([-1,1]),(np.arange(7)-3)/2))),list(np.concatenate((np.array([4/9,4/9]),np.repeat(1/10000,7))))),
                "Asymmetric claw":(list(np.concatenate((np.array([1/2]),(2**(1-np.arange(-2.0,3.0))/31)))) , list(np.concatenate((np.array([0]),np.arange(-2,3)+1/2))) , list(np.concatenate((np.array([1]),(2**-np.arange(-2.0,3.0)/10)**2))) ),
                "Asymmetric double claw":([46/100,46/100,1/300,1/300,1/300,7/300,7/300,7/300] , [-1,1,-1/2,-1,-3/2,1/2,1,3/2] , [4/9,4/9,(1/100)**2,(1/100)**2,(1/100)**2,(7/100)**2,(7/100)**2,(7/100)**2] ),
                "Smooth comb":(list(2**(5.0-np.arange(6))/63) , list((65-96*(1/2)**np.arange(6))/21) , list((32/63)**2/(2**(2*np.arange(6))))),
                "Discrete comb":(list(np.concatenate((np.repeat(2/7,3),np.repeat(1/21,3)))) , list(np.concatenate(((12*np.arange(3)-15)/7 , 2*np.arange(8,11)/7 ))) , list(np.concatenate((np.repeat(2/7,3)**2,np.repeat(1/21,3)**2))) )
                }
        self.upper = upper
        self.lower = lower
        self.inter = (upper-lower)/numbin
        self.ratio = shrink
        self.densityseq = None
        if idx is None:
            print("You did not specify the sample support, the default value is set to -4 to 4 with a precision of 0.04")
            self.idx = np.linspace(-4,4,200)
        else:
            self.idx = idx
    
    def MixSample(self, nsample,  weightlist, meanlist, stdlist):
        
        if not (len(weightlist) == len(meanlist) and len(meanlist) == len(stdlist)):
            raise ValueError("Check list length consistency")
        
        if abs(sum(np.array(weightlist))-1) > 0.00001:
            raise ValueError("Check your weight list, it does not sum up to 1")
        
        DataArray = np.empty(nsample)
        for i in range(nsample):
            ind = np.random.choice(np.arange(len(weightlist)), p = weightlist)
            DataArray[i] = np.random.normal(loc = meanlist[ind], scale = stdlist[ind])
        
        return DataArray
    
    def DynamicWindow(self):
        
        Allframes = np.arange(self.nframe)
        breakpoints = list(np.random.choice(np.arange(1,self.nframe),size = (self.ndensity-2), replace = False))
        breakpoints.sort()
        
        framelist = [Allframes[0:breakpoints[0]]]
        for i in range(self.ndensity-3):
            framelist.append(Allframes[breakpoints[i]:breakpoints[i+1]])
        framelist.append(Allframes[breakpoints[-1]:])
        
        LengthArray = np.empty(len(framelist))
        for i in range(len(framelist)):
            LengthArray[i] = len(framelist[i])
            
        return framelist, LengthArray
    
    def create_bin(self, obs):
        import numpy as np
        ind = np.arange(self.lower,self.upper+self.inter,self.inter)
        num = len(ind)
        rec = np.zeros(num - 1)
        for j in range(num-1):
            temp = (obs > ind[j]) & (obs < ind[j+1])
            count = np.count_nonzero(temp)
            rec[j] = count
        rec /= sum(rec)
        return rec
    
    def GMM_pdf(self,data, weight = None, mean = None,std = None, ind = 0):
        
        import numpy as np
        import scipy.stats as stat
        
        if weight is None:
            density = list(self.densitydict.values())[ind]
            weight = density[0]
            mean = density[1]
            std = np.sqrt(density[2])
        
        pdf = np.zeros(len(data))
        for w,m,s in zip(weight,mean,std):
            pdf += w*stat.norm.pdf(data,m,s)
        
        return pdf
    

    def Generation(self, nsamplelist,limit=True):
        
        datalist = []
        
        densitylist = list(self.densitydict.values())
        weightlist = []
        meanlist = []
        stdlist = []
        for i in range(len(densitylist)):
            weightlist.append(densitylist[i][0])
            meanlist.append(densitylist[i][1])
            stdlist.append(np.sqrt(densitylist[i][2]))
        
        framelist, lengtharray = self.DynamicWindow()
        
        indlist = []
        for i in range(len(lengtharray)):
            indlist.append(np.repeat(i,lengtharray[i]))
        
        indlist = np.hstack(indlist)
        self.densityseq = []

        for i in range(self.nframe-1):
            ind = indlist[i]
            weight1 = np.array(weightlist[ind])
            weight2 = np.array(weightlist[ind+1])
            actualweight = np.concatenate(((1-(i-sum(lengtharray[:ind]))/lengtharray[ind])*weight1,(i-sum(lengtharray[:ind]))/lengtharray[ind]*weight2))
            actualmean = np.concatenate((np.array(meanlist[ind]),np.array(meanlist[ind+1])))
            actualstd = np.concatenate((np.array(stdlist[ind]),np.array(stdlist[ind+1])))
            self.densityseq.append(self.GMM_pdf(self.idx,weight = actualweight, mean = actualmean, std = actualstd))
            datalist.append(self.MixSample(nsamplelist[i],actualweight,actualmean,actualstd))
            
        datalist.append(self.MixSample(nsamplelist[-1],weightlist[-1],meanlist[-1],stdlist[-1]))
        self.densityseq.append(self.GMM_pdf(self.idx, weight = weightlist[-1], mean = meanlist[-1], std = stdlist[-1]))
        
        if limit:
            upmax = 0
            lowmin = 0
            center = (self.upper+self.lower)/2
            span = self.upper-self.lower
            # for i in range(len(datalist)):
            #     if upmax < max(datalist[i]):
            #         upmax = max(datalist[i])
            #     if lowmin > min(datalist[i]):
            #         lowmin = min(datalist[i])
            maxran = 8
            self.idx = self.idx*self.ratio*span/maxran + center
            for i in range(len(datalist)):
                datalist[i] = datalist[i]*self.ratio*span/maxran + center
        return datalist

def split(data, p):
    import numpy as np
    import numpy.random as random
    import math
    temp1 = []
    temp2 = []
    for i in range(len(data)):
        n = math.floor(len(data[i])*p)
        ind = random.choice(np.arange(0,len(data[i])),n,replace = False)
        train = np.delete(data[i],ind)
        test = data[i][ind]
        temp1.append(train)
        temp2.append(test)
    return temp1,temp2

def burnin(data,n):
    return data[:n],data[n:]

def mean_size(obs,frame):
    import numpy as np
    temp = []
    n = int(max(obs[:,0])/frame)
    for i in range(0,n+1):
        ind = (obs[:,0] > (i*frame)) & (obs[:,0] <= ((i+1)*frame))
        size = obs[ind,1]
        size = size/np.mean(size)
        temp.append(size)
    if len(temp[n]) == 0:
        temp.pop()
    return temp

def nor(obs):
    temp = []
    for i in range(len(obs)):
        y = obs[i]/sum(obs[i])
        temp.append(y)
    return temp

def create_bin(obs, lower, upper, inter):
    import numpy as np
    ind = np.arange(lower,upper+inter,inter)
    n = len(obs)
    num = len(ind)
    lst = []
    for i in range(n):
        rec = np.zeros(num - 1)
        for j in range(num-1):
            temp = (obs[i] > ind[j]) & (obs[i] < ind[j+1])
            count = np.count_nonzero(temp)
            rec[j] = count
        lst.append(rec/sum(rec))
    return lst
            
        