class TAKDE():
    
    def __init__(self,upper = 2,lower = 0,inter = 0.1,cutoff = 0.5,alpha = 0.9):
        self.upper = upper
        self.lower = lower
        self.inter = inter
        self.memory = alpha
        self.cut = cutoff
        
    def create_bin(self, obs):
        import numpy as np
        ind = np.arange(self.lower,self.upper+self.inter,self.inter)
        num = len(ind)
        rec = np.zeros(num - 1)
        for j in range(num-1):
            temp = (obs > ind[j]) & (obs < ind[j+1])
            count = np.count_nonzero(temp)
            rec[j] = count
        rec /= (np.mean(rec)/(1/(self.upper-self.lower)))
        return rec
    
    def SinglePointKDE(self,test_point,x_base,width = 0):
        import numpy as np
        if width == 0:
            width = self.ref_width
        return max(sum(np.exp(-(x_base-test_point)**2/(2*width**2)))/(len(x_base)*np.sqrt(2*np.pi)*width),10**(-10))
    
    def MultiplePointKDE(self,test_points,x_base,width = 0):
        import numpy as np
        if width == 0:
            width = self.ref_width
        y = np.zeros(len(test_points))
        for i in range(len(test_points)):
            y[i] = self.SinglePointKDE(test_points[i],x_base,width)
        return y
    
    def widthcreator(self,train_list,rule="normal",cv = 0):
        
        import numpy as np
        if rule=="normal":
            constant = (32/3)**(1/5)
        elif rule=="oversmooth":
            constant = (972/(35*np.sqrt(np.pi)))**(1/5)
        elif rule=="cv":
            if cv==0:
                raise ValueError("Dude you need to input a cv parameter for the width generation")
            else:
                constant=cv
        else:
            constant=1
        
        width_list=[]
        for i in range(len(train_list)):
            temp = np.std(train_list[i])*constant/(len(train_list[i])*(2*len(train_list)-1))**(1/5)
            width_list.append(temp)
        
        return width_list
    
    def windowcreator(self,train_list,mass_list,t=10,fixedwindow=False):
        import numpy as np
        if not len(train_list)==len(mass_list):
            raise ValueError("trainlist length does not equal to mass list")
        window = 0
        dist = 0
        rb = []
        if fixedwindow:
            window = min(len(mass_list),t)
            for i in range(window):
                temp = np.mean(abs(mass_list[-1]-mass_list[-1-i])**2)
                rb.append(temp)
        else:
            while window <= (len(train_list)-1):
                temp = np.mean(abs(mass_list[-1]-mass_list[-1-window])**2)
                rb.append(temp)
                dist += temp
                window += 1
                if dist>=self.cut:
                    break
        rb.reverse()
        return window,rb
    
    def weightcreator(self,train_list,width_list,rb,mass_list = 0, weighting="amise"):
        
        if not len(train_list)==len(width_list):
            raise ValueError("Error when generating weights, number of widths does not equal to number of frames")
        
        import numpy as np
        weight_list = np.empty(len(train_list))
        
        if weighting=="amise":
            for i in range(len(train_list)):
                temp = 1.25/(2*np.sqrt(np.pi)*len(train_list[i])*width_list[i])+(2*len(train_list)-1)*rb[i]
                weight_list[i] = 1/temp
            weight_list /= sum(weight_list)
        
        elif weighting=="distance":
            if not isinstance(mass_list,list):
                raise ValueError("When using distance weighting scheme, you need to provide histogram mass_list to continue")
            for i in range(len(train_list)):
                weight_list[i] = 1/(sum(abs(mass_list[i]-mass_list[-1]))+1)
            weight_list /= sum(weight_list)
            
        elif weighting=="exponential":
            for i in range(len(train_list)):
                weight_list[i] = (1-self.memory)*self.memory**(len(train_list)-i-1)
            weight_list[0] /= 1-self.memory
            
        elif weighting=="average":
            for i in range(len(train_list)):
                weight_list[i] = 1/len(train_list)
        
        return weight_list
    
    def Estimation(self,train_list,test_points,prewidth,preweight):
        est = 0
        for i in range(len(train_list)):
            est += preweight[i]*self.MultiplePointKDE(test_points,train_list[i],prewidth[i])
        return est
    
    def Streaming_Estimation(self, train_list, test_list, width_list = 0, 
                             width_selector="normal", weighting = "amise",cv = 0,
                             fixedwindow=False,t=10):

        mass_list = []
        for i in range(len(train_list)):
            mass_list.append(self.create_bin(train_list[i]))   
            
        est = []
        if isinstance(width_list,list):
            est.append(self.MultiplePointKDE(train_list[0],test_list[0],width=width_list[0]))
            for i in range(1,len(train_list)):
                window,rb = self.windowcreator(train_list[:(i+1)],mass_list[:(i+1)],t=t,fixedwindow=fixedwindow)
                current_train = train_list[(i-window+1):(i+1)]
                local_width = width_list[(i-window+1):(i+1)]
                weight_list = self.weightcreator(current_train, rb = rb, width_list = local_width,mass_list=mass_list[(i-window+1):(i+1)],weighting=weighting)
                est.append(self.Estimation(current_train,test_list[i],prewidth=local_width,preweight=weight_list))
            
        else:
            width_list = self.widthcreator(train_list[:1],rule=width_selector,cv=cv)
            est.append(self.MultiplePointKDE(test_list[0],train_list[0],width=width_list[0]))
            for i in range(1,len(train_list)):
                window,rb = self.windowcreator(train_list[:(i+1)],mass_list[:(i+1)],t=t,fixedwindow=fixedwindow)
                current_train = train_list[(i-window+1):(i+1)]
                width_list = self.widthcreator(current_train,rule=width_selector,cv=cv)
                weight_list = self.weightcreator(current_train,rb = rb, width_list = width_list,mass_list=mass_list[(i-window+1):(i+1)],weighting=weighting)
                est.append(self.Estimation(current_train,test_list[i],prewidth=width_list,preweight=weight_list))

        return est
    
    def windowsizes(self, train_list,mass=None):
        import numpy as np
        if mass is None:
            mass_list = []
            for i in range(len(train_list)):
                mass_list.append(self.create_bin(train_list[i]))
        else:
            mass_list = mass
        window = np.zeros(len(train_list)-1)
        for i in range(1,len(train_list)):
            window[i-1],rb = self.windowcreator(train_list[:i],mass_list[:i])
        return window
    
    def width_updatetor(self,train_list,itr,width = 0,width_selector = 'normal',weighting = "amise"):
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
            
        else:
            temp = self.Streaming_Estimation(train_list,test_list,width_selector = width_selector,weighting = weighting)
            for i in range(len(temp)):
                y_spl = UnivariateSpline(itr,temp[i],s=0)
                y_spl_2d = y_spl.derivative(n=2)
                R = sum(y_spl_2d(itr)**2)/len(itr)
                width_update.append((4/(np.sqrt(np.pi)*len(train_list[i])*R))**0.2)
            
        return width_update


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

def standard(data,lower,upper,ratio):
    upmax = 0
    lowmin = 0
    span = upper - lower
    center = (upper + lower)/2
    actualcenter = 0
    tot = 0
    for i in range(len(data)):
        if upmax < max(data[i]):
            upmax = max(data[i])
        if lowmin > min(data[i]):
            lowmin = min(data[i])
        actualcenter += sum(data[i])
        tot += len(data[i])
    actualcenter /= tot
    maxran = upmax - lowmin
    temp = []
    for i in range(len(data)):
        temp.append((data[i]-actualcenter)*ratio*span/maxran + center)
    return temp

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

def splitvlist(data, p):
    import numpy as np
    import numpy.random as random
    temp1 = []
    temp2 = []
    for i in range(len(data)):
        n = p[i]
        ind = random.choice(np.arange(0,len(data[i])),n,replace = False)
        train = np.delete(data[i],ind)
        test = data[i][ind]
        temp1.append(train)
        temp2.append(test)
    return temp1,temp2

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
            
        rec /= (np.mean(rec)/(1/(upper-lower)))
        lst.append(rec/sum(rec))
    return lst