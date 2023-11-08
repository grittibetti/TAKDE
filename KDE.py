class KDEtrack():
    
    def __init__(self,grid,w,cv=(32/3)**(1/5)):
        self.grid = grid
        self.w = w
        self.cv = cv
        
    def SinglePointKDE(self,test_point,x_base,custom=0):
        import numpy as np
        if custom == 0:
            width = self.cv*np.std(x_base)*(1/len(x_base))**(1/5)
        else:
            width = custom
        return max(sum(np.exp(-(x_base-test_point)**2/(2*width**2)))/(len(x_base)*np.sqrt(2*np.pi)*width),10^(-10))
    
    def MultiplePointKDE(self,test_points,x_base,custom=0):
        import numpy as np
        y = np.zeros(len(test_points))
        for i in range(len(test_points)):
            y[i] = self.SinglePointKDE(test_points[i],x_base,custom=custom)
        return y
    
    def initial(self,obs):
        
        import numpy as np
        
        x_base = np.concatenate(obs[:self.w])
    
        est = self.MultiplePointKDE(self.grid,x_base)
        
        return est
        
    def resampling(self,grid,pd,obs,re = False):
        
        import numpy as np
        
        newpd = np.zeros(len(pd))
        newgrid = grid
        newdouble = np.zeros(len(pd))
        
        n1 = len(np.concatenate(obs[:len(obs)-1]))
        n2 = len(np.concatenate(obs[1:]))
        std1 = np.std(np.concatenate(obs[:len(obs)-1]))
        std2 = np.std(np.concatenate(obs[1:]))

        h1 = self.cv*(1/(len(np.concatenate(obs[:len(obs)-1]))))**(1/5)*std1
        h2 = self.cv*(1/(len(np.concatenate(obs[1:]))))**(1/5)*std2
        
        
        for i in range(len(newgrid)):
            newpd[i] = (pd[i] - self.SinglePointKDE(grid[i],obs[0],custom = h1)*len(obs[0])/n1)*n1/n2 + self.SinglePointKDE(grid[i],obs[-1],custom = h2)*len(obs[-1])/n2
            
        
        if re:
            
            for i in range(1,len(newdouble)-1):
                newdouble[i] = 2*((newpd[i+1]-newpd[i])/(newgrid[i+1]-newgrid[i])-(newpd[i]-newpd[i-1])/(newgrid[i]-newgrid[i-1]))/(newgrid[i+1]-newgrid[i-1])
            
            doublebar = np.mean(newdouble)
            
            ind_list = []
            grid_list = []
            pd_list = []
            
            for i in range(len(grid)-1):
                if max(newdouble[i],newdouble[i+1]) > doublebar:
                    ind_list.append(i+1)
                    grid_list.append((newgrid[i]+newgrid[i+1])/2)
                    pd_list.append((newpd[i]+newpd[i+1])/2)
            
            newgrid = np.insert(newgrid,ind_list,grid_list)
            newpd = np.insert(newpd,ind_list,pd_list)
            
            newdouble = np.zeros(len(newpd))
            
            for i in range(1,len(newdouble)-1):
                newdouble[i] = 2*((newpd[i+1]-newpd[i])/(newgrid[i+1]-newgrid[i])-(newpd[i]-newpd[i-1])/(newgrid[i]-newgrid[i-1]))/(newgrid[i+1]-newgrid[i-1])
            
            doublebar = np.mean(newdouble)
            
            ind_list = []
            
            for i in range(1,len(grid)-1):
                if max(newdouble[i-1],newdouble[i],newdouble[i+1]) < 0.05*doublebar:
                    ind_list.append(i)
            
            newgrid = np.delete(newgrid,ind_list)
            newpd = np.delete(newpd,ind_list)
            
        return newgrid,newpd
    
    def total_update(self,obs,re=False):
        
        
        grid = self.grid
        pd = self.initial(obs)
        
        grid_list = [grid]
        pd_list = [pd]
        
        for i in range(self.w,len(obs)):
            grid,pd = self.resampling(grid,pd,obs[i-self.w:i+1],re=re)
            
            grid_list.append(grid)
            pd_list.append(pd)
        
        return grid_list, pd_list
        
        
    def single_eval(self,x,pd,grid=None):
        import numpy as np
        
        if not len(pd)==len(grid):
            raise ValueError("grid list should has equal length to density list")
            
        if grid is None:
            grid = self.grid
            
        temp = abs(grid-x)
        ind = np.argsort(temp)[:2]
        D = temp[ind]
        d = pd[ind]
        est = max((D[0]*d[1]+D[1]*d[0])/sum(D),10**(-10))
        
        return est
        
    def multi_eval(self,x, pd, grid = None):
        
        import numpy as np
        
        if not len(pd)==len(grid):
            raise ValueError("grid list should has equal length to density list")
            
        if grid is None:
            grid = self.grid
        
        est = np.empty(len(x))
        
        for i in range(len(x)):
            temp = abs(grid-x[i])
            ind = np.argsort(temp)[:2]
            D = temp[ind]
            d = pd[ind]
            est[i] = max((D[0]*d[1]+D[1]*d[0])/sum(D),10**(-10))
        
        return est
    
    def Streaming(self,test,obs,re=False):
        
        if not (len(test) == len(obs)-self.w+1):
            raise ValueError("you test list length is in correct")
        else:
            grid, pd = self.total_update(obs,re=re)
            est = []
            
            for i in range(len(test)):
                est.append(self.multi_eval(test[i],pd[i],grid = grid[i]))
                
        return est
    
    