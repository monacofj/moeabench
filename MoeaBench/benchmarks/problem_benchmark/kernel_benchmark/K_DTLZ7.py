import numpy as np
from MoeaBench.H_DTLZ import H_DTLZ


class K_DTLZ7(H_DTLZ):
    
    def __init__(self, CACHE, **kwargs):
        self.CACHE=CACHE
        super().__init__(metodhs=set([1,2,4,5]),
                         **kwargs)


    def F1(self,I,M,X,GXm):
        return X[:,0:1]


    def F2(self,I,M,X,GXm):
        return X[:,I-1:I]


    def Fm1(self,I,M,X,GXm):
        return X[:,M-2:M-1]


    def Fm(self,I,M,X,GXm):
        return (1+GXm)*self.calc_H(M,X,GXm)


    def calc_F_M(self,Fi,M):
        if Fi == 1:
            return self.get_method(0)
        elif Fi >=2 and Fi <= M-2:
            return self.get_method(1)
        elif Fi > 1 and Fi == M-1:
            return self.get_method(2)
        elif Fi == M:
            return self.get_method(3)


    def calc_H(self,M,X,GXm):
        return M-np.sum(np.column_stack([X[:,Fi:Fi+1]/(1+GXm)*(1+np.sin(3*np.pi*X[:,Fi:Fi+1])) 
                                       for Fi in range(0,M-1)]), axis = 1).reshape(X.shape[0],1)    


    def param_F(self):
        dict_F = {
                    self.get_method(0) : self.F1,
                    self.get_method(1) : self.F2,
                    self.get_method(2) : self.Fm1,
                    self.get_method(3) : self.Fm
                  }
        return dict_F
   

    def calc_f(self,X,G):
        M = self.CACHE.get_BENCH_CI().get_M()
        vet_F_M = [self.calc_F_M(F,M) for F, i in enumerate(range(0,M), start = 1)]
        return np.column_stack(list(map(lambda Part: self.param_F()[Part[1]](Part[0],M,X,G),
                                        enumerate(vet_F_M, start = 1))))


    def calc_g(self,X):
        return 1+9/self.CACHE.get_BENCH_CI().get_K()*np.sum(X[:,self.CACHE.get_BENCH_CI().get_M()-1:], axis = 1).reshape(X.shape[0],1)

    
    def show_in(self,F):
        return {
           "IN POF"  : F                          
        }


    def minimize(self):
        X = self.get_Point_in_G()
        return self.show_in(self.calc_f(X,self.calc_g(X))),X
    

    def evaluation(self,x,n_ieq):  
        G=self.calc_g(x)
        F=self.calc_f(x,G)
        result =  {"F" : F} 
        return result





    
       
         