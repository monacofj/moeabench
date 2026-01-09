import numpy as np
from MoeaBench.H_DPF import H_DPF


class K_DPF1(H_DPF):
    
    def __init__(self, CACHE, **kwargs):
        self.CACHE=CACHE
        super().__init__(metodhs=set([4,5,7]),
                         methods_R1=set([]),
                         methods_R2=set([]),
                         **kwargs)


    def Y1 (self,D,X,Gxr):
       return 1/2*np.prod(X[:,0:D-1], axis = 1).reshape(X.shape[0],1)*(1+Gxr)


    def Y2 (self,D,X,Gxr):
        return 1/2*np.prod(X[:,0:D-2], axis = 1).reshape(X.shape[0],1)*(1-X[:,D-2:D-1])*(1+Gxr)


    def Yd (self,D,X,Gxr):
        return 1/2*(1-X[:,0:1])*(1+Gxr)
    

    def FD1(self,F,vet_chaos):
        return lambda row,D: np.dot(F,vet_chaos[row:row+D,:])
    

    def FM(self,F,vet_chaos):
        return lambda row,D: np.dot(F,vet_chaos[row:row+D,:])


    def calc_F_D(self,Fi,D):
         if Fi == 1:
             return self.get_method(0)
         elif Fi >= 2 and Fi <= D-1:
             return self.get_method(1)
         elif Fi == D:
             return self.get_method(2)


    def param_F(self):
        dict_PD = {
                    self.get_method(0)   : self.Y1,
                    self.get_method(1)   : self.Y2,
                    self.get_method(2)   : self.Yd
                  }
        return dict_PD
    

    def calc_F_PD(self,chaos,F,D,vet_chaos):
        redundat = []
        for index,row in enumerate(range(0,vet_chaos.shape[0],D), start = 0):
            redundat.append(chaos[index](row,D))
        return np.concatenate((F,np.column_stack(redundat)), axis = 1)


    def calc_f(self,X,G):
        D = self.CACHE.get_BENCH_CI().get_D()
        M = self.CACHE.get_BENCH_CI().get_M()
        vet_F_D = [self.calc_F_D(Fd,D) for Fd, i in enumerate(range(0,D), start = 1)]   
        F = np.column_stack(list(map(lambda Key: self.param_F()[Key](D,X,G),vet_F_D)))   
        vet_chaos = self.calc_NU(1,(M-D)*D,(M-D)*D)
        vet_F_C = [self.calc_F_C(Fc,M-D) for Fc, i in enumerate(range(0,M-D), start = 1)]
        chaos = list(map(lambda Keys: self.param_CHAOS()[Keys](F,vet_chaos),vet_F_C))
        return self.calc_F_PD(chaos,F,D,vet_chaos)
     

    def calc_g(self,X):
       D = self.CACHE.get_BENCH_CI().get_D()
       K = self.CACHE.get_BENCH_CI().get_K()
       return np.array([100*(K+np.sum(((Xi-0.5)**2) -  np.cos(20*np.pi*(Xi-0.5)))) for Xi in X[:,D-1:]]).reshape(X.shape[0],1)
       
     
    def minimize(self):
        X = self.get_Point_in_G()
        return self.show_in(self.calc_f(X,self.calc_g(X))),X
    

    def evaluation(self,x,n_ieq):  
        G=self.calc_g(x)
        F=self.calc_f(x,G)
        result =  {"F" : F} 
        if n_ieq != 0:       
            result["G"] = self.constraints(F)
            result["feasible"] = np.any((result["G"] <-0.00000000001)  | (result["G"] > 0.00000000001) )
        return result
      


      


  