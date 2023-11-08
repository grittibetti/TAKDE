def Reg(n):
    import numpy as np
    from numpy.linalg import pinv
    
    Cinv = np.zeros(shape=(n,n))
    
    for i in range(n):
        if (i%2 == 0):
            Cinv[1,i] = 1
        else:
            Cinv[0,i] = 1
        
        if i >= 2:
            Cinv[i,i-2] = -1
            Cinv[i,i-1] = 2
            Cinv[i,i] = -1
    
    C = pinv(Cinv)
    
    return C

class BK():
    
    def __init__(self,basis,n,lower = 0, upper = 2):
        self.b = basis
        self.n = n
        self.lower = lower
        self.upper = upper
        self.C = Reg(n)
        
    
    def create_bin(self, obs):
        import numpy as np
        inter = (self.upper-self.lower)/len(self.b)
        ind = np.arange(self.lower,self.upper+inter,inter)
        n = len(obs)
        num = len(ind)
        lst = []
        for i in range(n):
            rec = np.zeros(num - 1)
            for j in range(num-1):
                temp = (obs[i] > ind[j]) & (obs[i] < ind[j+1])
                count = np.count_nonzero(temp)
                rec[j] = count
            lst.append(rec)
        return lst
    
    
    def Bspline_basis(self,data = None):
        from scipy.interpolate import BSpline
        import numpy as np
        
        if data is None:
            data = self.b
        
        B = np.zeros(shape = (len(data),self.n))
        knots = np.linspace(self.lower,self.upper,self.n+1)
        space = (self.upper-self.lower)/self.n
        
        for i in range(len(knots)-1):
            tvec = [knots[i]-3*space,knots[i]-2*space,knots[i]-space]
            tvec.extend([knots[i],knots[i+1],knots[i+1]+space,knots[i+1]+2*space,knots[i+1]+3*space])
            B[:,i] = BSpline.basis_element(tvec,extrapolate=False)(data)
        
        B[np.isnan(B)] = 0
        
        return B
            
    
    def Bspline_Kalman(self,ini,noise_ini,obs,mtrx,cov,epsilon = 0.001,t = 10):
        import numpy as np
        import numpy.linalg as LA
        from numpy.linalg import pinv
        mtrxt = np.transpose(mtrx)
        update = []
        control = np.zeros(len(obs))
        update.append(ini)
        alpha = ini
        p = noise_ini
        for i in range(1,len(obs)):
            temp2 = np.dot(mtrx,alpha)
            counter = 0
            while True:
                alpha_hat = alpha
                a2 = alpha_hat
                p_hat = p + cov
                temp1 = np.dot(mtrx,alpha_hat)
                mu = obs[i] - temp1 -np.exp(-temp1)*(obs[i] - np.exp(temp1))
                H = np.diag(np.exp(-temp1))
                v = obs[i] - temp2 - mu
                F = LA.multi_dot([mtrx,p_hat,mtrxt]) + H
                K = LA.multi_dot([p_hat,mtrxt,pinv(F)])
                alpha_hat = alpha_hat + np.dot(K,v)
                counter += 1
                if LA.norm(a2-alpha_hat)<epsilon or counter > t:
                    break
            control[i] = LA.norm(mu)/LA.norm(obs[i])
            p = p_hat.dot(np.transpose(np.identity(len(ini)) - K.dot(mtrx)))
            alpha = alpha_hat
            update.append(alpha)
        return update
    
    def Reg(self):
        import numpy as np
        from numpy.linalg import pinv
        
        Cinv = np.zeros(shape=(self.n,self.n))
        
        for i in range(self.n):
            if (i%2 == 0):
                Cinv[1,i] = 1
            else:
                Cinv[0,i] = 1
            
            if i >= 2:
                Cinv[i,i-2] = -1
                Cinv[i,i-1] = 2
                Cinv[i,i] = -1
        
        C = pinv(Cinv)
        
        return C
    
    def Reg_BK(self,ini,noise_ini,obs,B,cov,epsilon = 0.001,t = 10):
        import numpy as np
        import numpy.linalg as LA
        from numpy.linalg import pinv
        
        C = self.Reg()
        mtrx = B@C
        mtrxt = np.transpose(mtrx)
        update = []
        control = np.zeros(len(obs))
        update.append(ini)
        alpha = ini
        p = noise_ini
        for i in range(1,len(obs)):
            temp2 = np.dot(mtrx,alpha)
            counter = 0
            while True:
                alpha_hat = alpha
                a2 = alpha_hat
                p_hat = p + cov
                temp1 = np.dot(mtrx,alpha_hat)
                mu = obs[i] - temp1 -np.exp(-temp1)*(obs[i] - np.exp(temp1))
                H = np.diag(np.exp(-temp1))
                v = obs[i] - temp2 - mu
                F = LA.multi_dot([mtrx,p_hat,mtrxt]) + H
                K = LA.multi_dot([p_hat,mtrxt,pinv(F)])
                alpha_hat = alpha_hat + np.dot(K,v)
                counter += 1
                if LA.norm(a2-alpha_hat)<epsilon or counter > t:
                    break
            control[i] = LA.norm(mu)/LA.norm(obs[i])
            p = p_hat.dot(np.transpose(np.identity(len(ini)) - K.dot(mtrx)))
            alpha = alpha_hat
            update.append(alpha)
        return update
    
    def point_eval(self,x,parameter,reg = True):
        
        import numpy as np
        
        data = [x]
        B = self.Bspline_basis(data=data)
        
        if reg:
            C = self.C
            Y = B@C@parameter
        
        else:
            Y = B@parameter
            
        return sum(np.exp(Y))
    
    def multi_eval(self,x,parameter,reg=True):
        
        import numpy as np
        from scipy.integrate import quad
        
        data = x
        B = self.Bspline_basis(data=data)
        
        if reg:
            Y = B@self.C@parameter
        
        else:
            Y = B@parameter
        
        area, dum = quad(self.point_eval,self.lower,self.upper,args = (parameter,reg))
        
        return np.exp(Y)/area
