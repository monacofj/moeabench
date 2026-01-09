import numpy as np
from itertools import combinations
from MoeaBench.H_DTLZ import H_DTLZ


class K_DTLZ8(H_DTLZ):

    def __init__(self, CACHE, **kwargs):
        self.CACHE=CACHE
        super().__init__(metodhs=set([6,7,8]),
                         **kwargs)


    def FiFj(self,I,N,M,X):
        return (1/(N/M))*np.sum((X[:,int((I-1)*(N/M)):int(I*(N/M))]),axis = 1).reshape(X.shape[0],1)
    

    def Gj(self,j,M,Fijx):
        return Fijx[:,M-1:M]+4*Fijx[:,j:j+1]-1


    def Gm(self,M,Fijx):
        return 2*Fijx[:,M-1:M]+self.calc_MinFiFj(M,Fijx)-1
   

    def calc_MinFiFj(self,M,Fijx):
        comb_sum_Fijx = list(combinations([MC for MC in range(0,M-1)],2))
        sum_cb = []
        sum_cb.extend(Fijx[:,cb[0]:cb[0]+1]+Fijx[:,cb[1]:cb[1]+1] for cb in comb_sum_Fijx)
        return np.min(np.column_stack(sum_cb), axis = 1).reshape(Fijx.shape[0],1)
    

    def param_F(self):
        dict_F = {
                    self.get_method(0) : self.FiFj,
                    self.get_method(1) : self.Gj,
                    self.get_method(2) : self.Gm
                 }
        return dict_F


    def calc_f(self,X,g=[]):
        M = self.CACHE.get_BENCH_CI().get_M()
        N = self.CACHE.get_BENCH_CI().get_Nvar()
        return np.column_stack(list(map(lambda Part: self.param_F()[Part[1]](Part[0],N,M,X),
                                        enumerate([self.get_method(0)for I in range(0,M)], start  = 1))))
    

    def calc_gijx(self,Fijx):
        M = self.CACHE.get_BENCH_CI().get_M()
        Gjx = np.column_stack(list(map(lambda Part: self.param_F()[Part[1]](Part[0],M,Fijx),
                                       enumerate([self.get_method(1) for I in range(0,M-1)], start  = 0))))
        Gmx = np.column_stack(list(map(lambda Key: self.param_F()[Key](M,Fijx),[self.get_method(2)])))
        return np.concatenate((Gjx,Gmx), axis = 1)

       

    def show_in(self,G,Fijx):
        condition = np.all(G >= 0,axis = 1)
        constraits_valid = Fijx[condition]  
        return {
           "IN POF"  : constraits_valid                          
        }


    def minimize(self):
        Xij = self.get_Point_in_G()
        Fijx = self.calc_f(Xij)
        return self.show_in(self.calc_gijx(Fijx),Fijx),Xij 
    

    def evaluation(self,X,n_ieq): 
        F=self.calc_f(X) 
        result = {"F" : F}
        if n_ieq != 0:
            G = self.calc_gijx(F)
            result['G']=-G
            result["feasible"] = np.all(result['G'] < 0)
        return result
        
        


        
