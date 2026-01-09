import numpy as np
from MoeaBench.H_DTLZ import H_DTLZ


class K_DTLZ3(H_DTLZ):

    def __init__(self, CACHE, **kwargs):
        self.CACHE=CACHE
        super().__init__(metodhs=set([1,2,3,5]),
                         **kwargs)


    def F1(self,M,X,Gxm):
        return (1+Gxm)*(np.prod(np.cos(X[:,:M-1]*np.pi/2), axis = 1).reshape(X.shape[0],1))


    def F2(self,M,X,Gxm):
        return (1+Gxm)*(np.prod(np.cos(X[:,:M-2]*np.pi/2),  axis = 1).reshape(X.shape[0],1))*np.sin(X[:,M-2:M-1]*np.pi/2)


    def F3(self,M,X,Gxm):
        return (1+Gxm)*(np.prod(np.cos(X[:,:M-3]*np.pi/2),  axis = 1).reshape(X.shape[0],1))*np.sin(X[:,M-3:M-2]*np.pi/2)


    def Fm(self,M,X,Gxm):
        return (1+Gxm)*np.sin(X[:,0:1]*np.pi/2)
    

    def param_F(self):
        dict_F = {
                    self.get_method(0) : self.F1,
                    self.get_method(1) : self.F2,
                    self.get_method(2) : self.F3,
                    self.get_method(3) : self.Fm
                  }
        return dict_F


    def calc_f(self,X,G):
        M = self.CACHE.get_BENCH_CI().get_M()
        vet_F_M = [self.calc_F_M(F,M) for F, i in enumerate(range(0,M), start = 1)]
        return np.column_stack(list(map(lambda Key: self.param_F()[Key](M,X,G),vet_F_M)))


    def calc_g(self,X):
         M = self.CACHE.get_BENCH_CI().get_M()
         K = self.CACHE.get_BENCH_CI().get_K()
         return np.array([100*(K+np.sum(((Xi-0.5)**2)-np.cos(20*np.pi*(Xi-0.5)))) for Xi in X[:,M-1:]]).reshape(X.shape[0],1)


    def minimize(self):
        X = self.get_Point_in_G()
        return self.show_in(self.eval_cons(self.calc_f(X,self.calc_g(X)))),X
    

    def evaluation(self,x,n_ieq):  
        G=self.calc_g(x)
        F=self.calc_f(x,G)
        result =  {"F" : F} 
        if n_ieq != 0:       
            result["G"] = self.constraints(F)
            result["feasible"] = np.any((result["G"] <-0.00000000001)  | (result["G"] > 0.00000000001) )
        return result
      


      