import numpy as np
from MoeaBench.H_DPF import H_DPF


class K_DPF5(H_DPF):

    def __init__(self, CACHE, **kwargs):
        self.CACHE=CACHE
        super().__init__(metodhs=set([1,2,3,4,5,6,7]),
                         methods_R1=set([1,2,3,5,6,7]),
                         methods_R2=set([4,5,6,7]),
                         **kwargs)
     

    def B1(self,D,X,M,GXr):
        return np.prod(np.cos(self.calc_TH(X[0:M-2])))*np.cos(self.calc_TH(X[M-2:M-1]))*(1+GXr)
    

    def B2(self,D,X,M,GXr):
        return np.prod(np.cos(self.calc_TH(X[0:M-2])))*np.sin(self.calc_TH(X[M-2:M-1]))*(1+GXr)
    

    def Bmd1(self,D,X,M,GXr):
        return np.prod(np.cos(self.calc_TH(X[0:D-1])))*np.sin(self.calc_TH(X[D-1:D]))*(1+GXr)
    

    def Y1(self,D,X,M,GXr):
        return np.sqrt(1/(M-D+1))*np.prod(np.cos(self.calc_TH(X[0:D-1])))*(1+GXr)
    

    def Y2(self,D,X,M,GXr):
        return np.prod(np.cos(self.calc_TH(X[0:D-2])))*np.sin(self.calc_TH(X[D-1:D]))*(1+GXr)


    def Yd1(self,D,X,M,GXr):
        return np.cos(self.calc_TH(X[0:1]))*np.sin(self.calc_TH(X[1:2]))*(1+GXr)
    

    def Yd(self,D,X,M,GXr):
        return np.sin(self.calc_TH(X[0:1]))*(1+GXr)


    def calc_F_P(self,D,M):   
        vet_FD=[]    
        for Fi,i in enumerate(range(0,M), start = 1):
            if Fi == 1:
                vet_FD.append(self.get_method_R1(0))
            elif Fi >= 2 and Fi < M-D+1:
                vet_FD.append(self.get_method_R1(1))
            elif Fi == M-D+1:
                vet_FD.append(self.get_method_R1(2))
            elif Fi >= M-D+1 and Fi < M-1:
                vet_FD.append(self.get_method_R1(3))
            elif Fi > M-D+1 and Fi == M-1:
                vet_FD.append(self.get_method_R1(4))
            elif Fi == M:
                vet_FD.append(self.get_method_R1(5))
        return tuple(vet_FD)


    def calc_F_D(self,D,M): 
        vet_FD=[]    
        for Fi,i in enumerate(range(0,M), start = 1):
            if Fi <= M-D+1:
                 vet_FD.append(self.get_method_R2(0))
            elif Fi > M-D+1 and Fi <= M-2:
                vet_FD.append(self.get_method_R2(1))
            elif Fi > M-D+1 and Fi == M-1:
                vet_FD.append(self.get_method_R2(2))
            elif Fi == M:
                vet_FD.append(self.get_method_R2(3))
        return tuple(vet_FD)
      

    def calc_F_PD(self,D,X,M):
        return [self.calc_F_D(D,M)  if Xi > 1/3 else self.calc_F_P(D,M)  for Xi in X[:,0:1]] 
    

    def param_F(self):
        dict_PD = {
                    self.get_method(0) : self.B1,
                    self.get_method(1) : self.B2,
                    self.get_method(2) : self.Bmd1,
                    self.get_method(3) : self.Y1,
                    self.get_method(4) : self.Y2,
                    self.get_method(5) : self.Yd1,
                    self.get_method(6) : self.Yd
                  }
        return dict_PD


    def calc_f(self,X,G):
       M = self.CACHE.get_BENCH_CI().get_M()
       D = self.CACHE.get_BENCH_CI().get_D()
       return np.array([list(map(lambda Keys: self.param_F()[Keys](D,X[i],M,G[i])[0],F_PD)) 
                        for i, F_PD in enumerate(self.calc_F_PD(D,X,M), start = 0)])

     
    def calc_g(self,X):
        M=self.CACHE.get_BENCH_CI().get_M()
        return ((X[:,M-1:M]-X[:,0:1])**2)+np.array([np.sum((Xi-0.5)**2) 
                                                    for Xi in X[:,M:]]).reshape(X.shape[0],1)


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
      

      
