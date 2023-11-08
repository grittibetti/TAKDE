# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:11:15 2021

@author: ronny
"""

#%% 

class Measurement():
    
    def __init__(self, init_data,upper = 2,lower = 0,inter = 0.1):
        self.upper = upper
        self.lower = lower
        self.inter = inter
        self.data = init_data
        
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
    
    def FixedWidth(self,train_points):
        return 1/len(train_points)
    
    def NoMetric(self,train_points):
        return 1
    
    def NormalRule(self,train_points):
        import numpy as np
        return 1/(np.std(train_points))**5
    
    def SecondOrderIntegration(self,train_points):
        hist_weights = self.create_bin(train_points)
        approx = 0
        for i in range(1,len(hist_weights)-1):
            approx += (hist_weights[i-1]+hist_weights[i+1]-2*hist_weights[i])**2
        approx /= self.inter**4*len(hist_weights)
        return approx

class TAKDE(Measurement):
    
    def __init__(self,init_data ,width = 1,upper = 2,lower = 0,inter = 0.1,cutoff = 2,metric = "FixedWidth",alpha = 0.9):
        super().__init__(upper,lower,inter)
        self.data = init_data
        self.ref_width = width
        self.ref_nod = len(init_data)
        self.upper = upper
        self.lower = lower
        self.inter = inter
        self.memory = alpha
        self.func = metric
        self.cut = cutoff
    
    def SinglePointKDE(self,test_point,x_base,width = 0):
        import numpy as np
        if width == 0:
            width = self.ref_width
        return max(sum(np.exp(-(x_base-test_point)**2/(2*width**2)))/(len(x_base)*np.sqrt(2*np.pi)*width),10^(-10))
    
    def MultiplePointKDE(self,test_points,x_base,width = 0):
        import numpy as np
        if width == 0:
            width = self.ref_width
        y = np.zeros(len(test_points))
        for i in range(len(test_points)):
            y[i] = self.SinglePointKDE(test_points[i],x_base,width)
        return y
    
    def Estimation(self,train_list,test_points,prewidth = 0):
        if isinstance(prewidth,list):
            est = 0
            for i in range(len(train_list)-1):
                est += (1-self.memory)*self.memory**i*self.MultiplePointKDE(test_points,train_list[-(i+1)],prewidth[-(i+1)])
            est += self.memory**(len(train_list)-1)*self.MultiplePointKDE(test_points,train_list[0],prewidth[0])
        else:
            met = Measurement(init_data=self.data,upper = self.upper, lower = self.lower, inter = self.inter)
            m_init = getattr(met,self.func)(self.data)
            est = 0
            for i in range(len(train_list)-1):
                m_current = getattr(met,self.func)(train_list[-(i+1)])
                width_current = self.ref_width*(self.ref_nod*m_init/(len(train_list[-(i+1)])*m_current))**(1/5)
                est += (1-self.memory)*self.memory**i*self.MultiplePointKDE(test_points,train_list[-(i+1)],width_current)
            m_current = getattr(met,self.func)(train_list[0])
            width_current = self.ref_width*(self.ref_nod*m_init/(len(train_list[0])*m_current))**(1/5)
            est += self.memory**(len(train_list)-1)*self.MultiplePointKDE(test_points,train_list[0],width_current)
        return est
    
    def Streaming_Estimation(self, train_list, test_list, width_list = 0, weighting = "distance", normalizer = 1):
        import numpy as np
        mass_list = []
        for i in range(len(train_list)):
            mass_list.append(self.create_bin(train_list[i]))
        met = Measurement(init_data=self.data,upper = self.upper, lower = self.lower, inter = self.inter)
        m_init = getattr(met,self.func)(self.data)
        est = []
        est.append(self.MultiplePointKDE(test_list[0],train_list[0],self.ref_width))
        if isinstance(width_list,list):
            width_list = width_list
        else:
            width_list = [self.ref_width]
            for i in range(1,len(train_list)):
                m_current = getattr(met,self.func)(train_list[i])
                width_list.append(self.ref_width*(self.ref_nod*m_init/(len(train_list[i])*m_current))**(1/5))
        if weighting == "distance":
            for i in range(1,len(train_list)):
                window = 0
                dist = 0
                weight_list = []
                while window <= i:
                    dist += sum(abs(mass_list[i] - mass_list[i-window]))
                    weight_list.append(1/(sum(abs(mass_list[i] - mass_list[i-window]))+normalizer))
                    window += 1
                    if dist >= self.cut:
                        break
                softmax_total = 0
                for weight in weight_list:
                    softmax_total += weight
                temp = 0
                for j in range(window):
                    temp += weight_list[j]/softmax_total*self.MultiplePointKDE(test_list[i],train_list[i-j],width_list[i-j])
                est.append(temp)
        elif weighting == "exponential":
            for i in range(1,len(train_list)):
                window = 0
                dist = 0
                weight_list = []
                while window <= i:
                    dist += sum(abs(mass_list[i] - mass_list[i-window]))
                    weight_list.append((1-self.memory)*self.memory**window)
                    window += 1
                    if dist >= self.cut:
                        break
                weight_list[-1] /= (1-self.memory)
                temp = 0
                for j in range(window):
                    temp += weight_list[j]*self.MultiplePointKDE(test_list[i], train_list[i-j],width_list[i-j])
                est.append(temp)
        elif weighting == "average":
            for i in range(1,len(train_list)):
                window = 0
                dist = 0
                while window <= i:
                    dist += sum(abs(mass_list[i] - mass_list[i-window]))
                    window += 1
                    if dist >= self.cut:
                        break
                weight_list = np.ones(window)/window
                temp = 0
                for j in range(window):
                    temp += weight_list[j]*self.MultiplePointKDE(test_list[i], train_list[i-j],width_list[i-j])
                est.append(temp)
        else:
            for i in range(1,len(train_list)):
                window = 0
                dist = 0
                weight_list = []
                while window <= i:
                    dist += sum(abs(mass_list[i] - mass_list[i-window]))
                    weight_list.append((1-self.memory)*self.memory**window/(sum(abs(mass_list[i] - mass_list[i-window]))+normalizer))
                    window += 1
                    if dist >= self.cut:
                        break
                weight_list[-1] /= (1-self.memory)
                softmax_total = 0
                for weight in weight_list:
                    softmax_total += weight
                temp = 0
                for j in range(window):
                    temp += weight_list[j]/softmax_total*self.MultiplePointKDE(test_list[i],train_list[i-j],width_list[i-j])
                est.append(temp)
        return est
      
    def width_updatetor(self,train_list,itr,width = 0,weighting = "distance"):
        import numpy as np
        from scipy.interpolate import UnivariateSpline
        test_list = [itr] * len(train_list)
        width_update = []
        if isinstance(width,list):
            temp = self.Streaming_Estimation(train_list,test_list,width_list = width,weighting = weighting)
            for i in range(len(temp)):
                y_spl = UnivariateSpline(itr,temp[i],s=0)
                y_spl_2d = y_spl.derivative(n=2)
                R = sum(y_spl_2d(itr)**2)/len(itr)
                width_update.append((4/(np.sqrt(np.pi)*len(train_list[i])*R))**0.2)
            width_update.insert(0,width[0])
        else:
            temp = self.Streaming_Estimation(train_list,test_list,weighting = weighting)
            for i in range(len(temp)):
                y_spl = UnivariateSpline(itr,temp[i],s=0)
                y_spl_2d = y_spl.derivative(n=2)
                R = sum(y_spl_2d(itr)**2)/len(itr)
                width_update.append((4/(np.sqrt(np.pi)*len(train_list[i])*R))**0.2)
            width_update.insert(0,width)
        return width_update
    
    def windowsizes(self, train_list):
        mass_list = []
        for i in range(len(train_list)):
            mass_list.append(self.create_bin(train_list[i]))
        size_list = []
        for i in range(1,len(train_list)):
            dist = 0
            window = 0
            while window <= i:
                dist += sum(abs(mass_list[i] - mass_list[i-window]))
                window += 1
                if dist >= self.cut:
                    break
            size_list.append(window)
        return size_list
        

def Kalman(ini,noise_ini,obs,kernel,cov,epsilon = 0.001,t = 10):
    import numpy as np
    import numpy.linalg as LA
    from numpy.linalg import pinv
    kernelt = np.transpose(kernel)
    update = []
    control = np.zeros(len(obs))
    update.append(ini)
    alpha = ini
    p = noise_ini
    for i in range(1,len(obs)):
        temp2 = np.dot(kernel,alpha)
        counter = 0
        while True:
            alpha_hat = alpha
            a2 = alpha_hat
            p_hat = p + cov
            temp1 = np.dot(kernel,alpha_hat)
            mu = obs[i] - temp1 -np.exp(-temp1)*(obs[i] - np.exp(temp1))
            H = np.diag(np.exp(-temp1))
            v = obs[i] - temp2 - mu
            F = LA.multi_dot([kernel,p_hat,kernelt]) + H
            K = LA.multi_dot([p_hat,kernelt,pinv(F)])
            alpha_hat = alpha_hat + np.dot(K,v)
            counter += 1
            if LA.norm(a2-alpha_hat)<epsilon or counter > t:
                break
        control[i] = LA.norm(mu)/LA.norm(obs[i])
        p = p_hat.dot(np.transpose(np.identity(len(ini)) - K.dot(kernel)))
        alpha = alpha_hat
        update.append(alpha)
    return update,control

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

def burnin(data,n):
    return data[:n],data[n:]

def noise_est(data,k,kb,kernel,n,q=0.04,p=0.02,a=1,b=1):
    import numpy as np
    import scipy.stats as stat
    t = len(data)
    Q = q*np.identity(n)
    q_vec = np.zeros(k)
    a0 = np.ones(n)
    w0 = 0.01*np.ones(n)
    gamma_pre = Kalman(a0,w0,data,kernel,Q,0.01,5)
    a += (t-1)*n/2
    b_rec = np.zeros(k)
    for i in range(k):
        gamma = gamma_pre
        dif = np.diff(gamma_pre,axis = 0)
        # dif = np.delete(dif,[0,len(dif)-1],axis = 0)
        b1 = b+sum(sum(dif**2))/2                               #Update inverse gamma parameter
        q = stat.invgamma.rvs(a,scale = b1)                                             #update estimation by inverse gamma sampling
        for j in range(1,t-1):
            gamma[j] = stat.multivariate_normal.rvs(mean = gamma_pre[j],cov = p)        #generate new gamma from normal (gamma,R)
            pos = np.prod(stat.poisson.pmf(data[j],np.dot(kernel,gamma[j])))            #Poisson likelihood at k
            pos_pre = np.prod(stat.poisson.pmf(data[j],np.dot(kernel,gamma_pre[j])))    #Poisson likelihood at k-1
            nor1 = stat.multivariate_normal.pdf(gamma[j],mean = gamma[j-1],cov = q)           #Normal likelihood at t-1/k and t/k
            nor2 = stat.multivariate_normal.pdf(gamma_pre[j+1],mean = gamma[j],cov = q)       #Normal likelihood at t+1/k-1 and t/k
            nor3 = stat.multivariate_normal.pdf(gamma_pre[j],mean = gamma[j-1],cov = q)       #Normal likelihood at t/k-1 and t-1/k
            nor4 = stat.multivariate_normal.pdf(gamma_pre[j+1],mean = gamma_pre[j],cov = q)   #Normal likelihood at t/k-1 and t+1/k-1
            r = pos*nor1*nor2/(pos_pre*nor3*nor4)
            if r < stat.uniform.rvs():
                gamma[j] = gamma_pre[j]
        gamma_pre = gamma
        q_vec[i] = q
        b_rec[i] = b1
    return np.mean(q_vec[kb:]),b_rec

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


            
 