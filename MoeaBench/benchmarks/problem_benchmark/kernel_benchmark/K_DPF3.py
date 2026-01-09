import numpy as np
from MoeaBench.H_DPF import H_DPF


class K_DPF3(H_DPF):

     def __init__(self, CACHE, **kwargs):
        self.CACHE=CACHE
        super().__init__(metodhs=set([4,5,6,7]),
                         methods_R1=set([8]),
                         methods_R2=set([]),
                         **kwargs)


     def Y1 (self,D,X,GXr):
        return (1-np.prod(np.cos(self.calc_TH(X[:,0:D-1],100)), axis = 1).reshape(X.shape[0],1))*(1+GXr)


     def Y2 (self,D,X,GXr):
        return (1-np.prod(np.cos(self.calc_TH(X[:,0:D-2],100)), axis = 1)
                .reshape(X.shape[0],1)*np.sin(self.calc_TH(X[:,D-2:D-1],100)))*(1+GXr)


     def Yd1 (self,D,X,GXr):
        return (1-np.cos(self.calc_TH(X[:,0:1],100))*np.sin(self.calc_TH(X[:,1:2],100)))*(1+GXr)


     def Yd (self,D,X,GXr):
        return (1-np.sin(self.calc_TH(X[:,0:1],100)))*(1+GXr)
   

     def FD(self,Yd,vet_chaos):
        return lambda row,col,col_next: min(Yd[row],vet_chaos[row][0])
    

     def FD1(self,Yd,vet_chaos):
        return lambda row,col,col_next: min(max(Yd[row],vet_chaos[row][col]),vet_chaos[row][col_next])
     

     def FM(self,Yd,vet_chaos):
        return lambda row,col,col_next: max(Yd[row],vet_chaos[row][col])
     

     def calc_f(self,X,G): 
         M = self.CACHE.get_BENCH_CI().get_M()
         D = self.CACHE.get_BENCH_CI().get_D()
         vet_F_D = [self.calc_F_D(Fd,D) for Fd, i in enumerate(range(0,D), start = 1)]  
         Yd1 = np.column_stack(list(map(lambda Keys: self.param_F()[Keys](D,X,G),vet_F_D[:-1])))
         Yd = np.column_stack(list(map(lambda Keys: self.param_F()[Keys](D,X,G),vet_F_D[D-1:D])))     
         vet_chaos = np.array((self.calc_NU(M-D,X.shape[0])))
         vet_chaos = np.insert(vet_chaos,0,vet_chaos[:,0], axis =1)
         vet_F_C = [self.calc_F_C(Fc,M-D) for Fc, i in enumerate(range(0,M-D), start = 1)]
         vet_F_C.insert(0,self.get_method_R1(0))
         chaos = list(map(lambda Keys: self.param_CHAOS()[Keys](Yd,vet_chaos),vet_F_C))
         return self.calc_F_PD(X,chaos,Yd1,vet_chaos)
                             

     def calc_g(self,X):
         D = self.CACHE.get_BENCH_CI().get_D()
         return np.array([np.sum(((Xi-0.5)**2)) for Xi in X[:,D-1:]]).reshape(X.shape[0],1)


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
      


      