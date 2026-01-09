import numpy as np
from MoeaBench.H_DPF import H_DPF

class K_DPF2(H_DPF):
   
   
   def __init__(self, CACHE, **kwargs):
        self.CACHE=CACHE
        super().__init__(metodhs=set([4,5,6,7]),
                         methods_R1=set([]),
                         methods_R2=set([]),
                         **kwargs)


   def Y1 (self,D,X,GXr,ind_Y2_D1):
        return X[:,0:1]


   def Y2 (self,D,X,GXr,ind_Y2_D1):
        return X[:,ind_Y2_D1-1:ind_Y2_D1]


   def Yd1 (self,D,X,GXr,ind_Y2_D1):
        return X[:,ind_Y2_D1-1:ind_Y2_D1]


   def Yd (self,D,X,GXr,ind_Y2_D1):
        return (D-np.sum(X[:,0:D-1]/(1+GXr)*(1+np.sin(3*np.pi*X[:,0:D-1])), axis = 1).reshape(X.shape[0],1))*(1+GXr)
     

   def FD1(self,F,vet_chaos):
        return lambda row,D: np.dot(F,vet_chaos[row:row+D,:])**2
    

   def FM(self,F,vet_chaos):
        return lambda row,D: np.dot(F,vet_chaos[row:row+D,:])**2
     

   def calc_F_PD(self,chaos,F,D,vet_chaos):
         redundat = []
         for index,row in enumerate(range(0,vet_chaos.shape[0],D), start = 0):
             redundat.append(chaos[index](row,D))
         return np.concatenate((F,np.column_stack(redundat)), axis = 1)
     

   def calc_f(self,X,G): 
         D = self.CACHE.get_BENCH_CI().get_D()
         M = self.CACHE.get_BENCH_CI().get_M()
         vet_F_D = [self.calc_F_D(Fd,D) for Fd, i in enumerate(range(0,D), start = 1)]   
         F = np.column_stack(list(map(lambda Part: self.param_F()[Part[1]](D,X,G,Part[0]),
                                      enumerate(vet_F_D, start  = 1))))    
         vet_chaos = self.calc_NU(1,(M-D)*D,(M-D)*D)
         vet_F_C = [self.calc_F_C(Fc,M-D) for Fc, i in enumerate(range(0,M-D), start = 1)]
         chaos = list(map(lambda Keys: self.param_CHAOS()[Keys](F,vet_chaos),vet_F_C))
         return self.calc_F_PD(chaos,F,D,vet_chaos)
         

   def calc_g(self,X):
         D = self.CACHE.get_BENCH_CI().get_D()
         K = self.CACHE.get_BENCH_CI().get_K()
         return np.array([1+9/K*np.sum(Xi) for Xi in X[:,D-1:]]).reshape(X.shape[0],1)
     

   def minimize(self):
         X = self.get_Point_in_G()
         return self.show_in(self.calc_f(X,self.calc_g(X))),X
   

   def evaluation(self,x,n_ieq):  
        G=self.calc_g(x)
        F=self.calc_f(x,G)
        result =  {"F" : F} 
        return result
                
      

      
    