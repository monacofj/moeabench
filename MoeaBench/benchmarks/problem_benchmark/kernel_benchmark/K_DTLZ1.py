import numpy as np
from MoeaBench.H_DTLZ import H_DTLZ


class K_DTLZ1(H_DTLZ):

    def __init__(self, CACHE, **kwargs):
        self.CACHE=CACHE
        super().__init__(metodhs=set([1,2,4,5]),
                         **kwargs)

                 
    def F1(self,M,X,Gxm):
        return 1/2*np.prod(X[:,:M-1], axis = 1).reshape(X.shape[0],1)*(1+Gxm)


    def F2(self,M,X,Gxm):
        return 1/2*np.prod(X[:,:M-2], axis = 1).reshape(X.shape[0],1)*(1-X[:,M-2:M-1])*(1+Gxm)


    def Fm1(self,M,X,Gxm):
        return 1/2*X[:,0:1]*(1-X[:,1:2])*(1+Gxm)


    def Fm(self,M,X,Gxm):
        return 1/2*(1-X[:,0:1])*(1+Gxm)
 
 
    def calc_F_M(self,Fi,M):
        if Fi == 1:
            return self.get_method(0)
        elif Fi >=2 and Fi <= M-2:
            return self.get_method(1)
        elif Fi > 1 and Fi == M-1:
            return self.get_method(2)
        elif Fi == M:
            return self.get_method(3)


    def param_F(self):
        dict_F = {
                    self.get_method(0)  : self.F1,
                    self.get_method(1)  : self.F2,
                    self.get_method(2)  : self.Fm1,
                    self.get_method(3)  : self.Fm
                  }
        return dict_F


    def calc_f(self,X,G):
        M = self.CACHE.get_BENCH_CI().get_M()
        vet_F_M = [self.calc_F_M(F,M) for F, i in enumerate(range(0,M), start = 1)]
        return np.column_stack(list(map(lambda Key: self.param_F()[Key](M,X,G),vet_F_M)))
     

    def calc_g(self,X):
        M = self.CACHE.get_BENCH_CI().get_M()
        K = self.CACHE.get_BENCH_CI().get_K()
        return 100*(K+np.sum(np.array([(((Xi-0.5)**2) - np.cos(20*np.pi*(Xi-0.5))) 
                                       for Xi in X[:,M-1:]]), axis = 1).reshape(X.shape[0],1))
                   

    def minimize(self):
        X = self.get_Point_in_G()
        return self.show_in(self.eval_cons(self.calc_f(X,self.calc_g(X)),False)),X
    

    def evaluation(self,x,n_ieq):  
        G=self.calc_g(x)
        F=self.calc_f(x,G)
        result =  {"F" : F} 
        if n_ieq != 0:       
            result["G"] = self.constraints(F)
            result["feasible"] = np.any((result["G"] <-0.00000000001)  | (result["G"] > 0.00000000001) )
        return result
      


      
    
        
    

    
                            

       


    



        

