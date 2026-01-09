from MoeaBench.base_benchmark import BaseBenchmark
from MoeaBench.integration_benchmark import integration_benchmark
from enum import Enum
import numpy as np


class my_dtlz5(integration_benchmark):
        
        def __init__(self, M = 3 ,P = 600 ,K = 5, D = 2, types = "IN POF"):
          super().__init__(dtlz5,types,M,P,K,D)
                       

class E_DTLZ(Enum):
       F1   = 1
       F2   = 2
       F3   = 3 
       Fm   = 5


class dtlz5(BaseBenchmark):

    def __init__(self, types : str = None, M : int = 3, P : int = 700, K : int = 10, N : int = 0, D : int = 2, n_ieq_constr : int = 1):
        super().__init__(types, M, P, K, self.calc_N, n_ieq_constr)
        self.llist_E_DTLZ = list(E_DTLZ)
        

    def calc_N(self,K,M):
        return K+M-1


    def constraits(self,f,parameter = 1,f_c=[]):
        f_constraits=np.array(f)
        f_c = np.array([np.sum([ f_c**2  for  f_c in f_constraits[linha,0:f_constraits.shape[1]]])-parameter for index,linha in enumerate(range(f_constraits.shape[0]))  ])
        return f_c


    def eval_cons(self,f):
        M_constraits = self.constraits(f)
        eval = M_constraits == 0
        return f[eval]


    def get_Points(self):
        return np.array([*np.random.random((self.get_P(), self.get_N()))*1.0])


    def F1(self,M,th,Gxm):
       theta = list(map(lambda TH: np.cos(TH), th[0:(M-1)]))
       return (1+Gxm)*np.prod(np.column_stack(theta ), axis = 1).reshape(Gxm.shape[0],1)


    def F2(self,M,th,Gxm):
        theta = list(map(lambda TH: np.cos(TH), th[0:(M-2)]))
        return (1+Gxm)*np.prod(np.column_stack(theta ), axis = 1).reshape(Gxm.shape[0],1)*np.column_stack(np.sin(th[(M-2):(M-1)]))


    def F3(self,M,th,Gxm):
        theta = list(map(lambda TH: np.cos(TH), th[0:(M-3)]))
        return (1+Gxm)*np.prod(np.column_stack(theta ), axis = 1).reshape(Gxm.shape[0],1)*np.column_stack(np.sin(th[(M-3):(M-2)]))


    def Fm(self,M,th,Gxm):
        return (1+Gxm)*np.column_stack(np.sin(th[0:1]))


    def get_method(self,enum):
        return self.llist_E_DTLZ[enum]


    def param_F(self):
        dict_F = {
                    self.get_method(0) : self.F1,
                    self.get_method(1) : self.F2,
                    self.get_method(2) : self.F3,
                    self.get_method(3) : self.Fm
                  }
        return dict_F


    def calc_F_M(self,Fi,M):
        if Fi == 1:
            return self.get_method(0)
        elif Fi == 2 and M > 2:
            return self.get_method(1)
        elif Fi >= 3 and Fi <= M-1 and M > 3:
            return self.get_method(2)
        elif Fi == M:
            return self.get_method(3)


    def calc_TH(self,X,Gxm,M):
        return [X[:,Xi:Xi+1]*np.pi/2 if Xi == 0 else (np.pi/(4*(1+Gxm))*(1+2*Gxm*X[:,Xi:Xi+1]))  for Xi in range(0,M-1)]


    def calc_f(self,X,G):
        vet_F_M = [self.calc_F_M(F,self.get_M()) for F, i in enumerate(range(0,self.get_M()), start = 1)]
        return np.column_stack(list(map(lambda Key: self.param_F()[Key](self.get_M(),self.calc_TH(X,G,self.get_M()),G),vet_F_M)))


    def calc_g(self,X):
        return np.sum((X[:,self.get_M()-1:]-0.5)**2, axis = 1).reshape(X.shape[0],1)
    
    
    def set_Point_in_G(self,X):
       self._point_in_g = X
    

    def get_Point_in_G(self):
       return self._point_in_g


    def POFsamples(self):
        X = self.get_Points()
        X[:,self.get_M()-1:self.get_N()]=0.5
        self.set_Point_in_G(X)
        G = self.calc_g(X)
        F = self.eval_cons(self.calc_f(self.get_Point_in_G(),G))
        return F


    def evaluation(self,x,n_ieq):
        G=self.calc_g(x)
        F=self.calc_f(x,G)
        result =  {"F" : F}
        if n_ieq != 0:
            cons = self.constraits(F,1.25)
            const  = cons.reshape(cons.shape[0],1)
            result["G"] = const
            result["feasible"] = np.any((result["G"] <-0.00000000001)  | (result["G"] > 0.00000000001) )
        return result



