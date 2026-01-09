from .constraits_1 import constraits_1
from .constraits_05 import constraits_05
from .I_constraints import I_constraints
from .evalue_constraits import evalue_constraits
import numpy as np
from .I_IBENCH import I_IBENCH
from.allowed_benchmark import allowed_benchmark


class InitBenchmark(evalue_constraits,constraits_1,constraits_05,I_constraints,I_IBENCH,allowed_benchmark):
       
    def __init__(self,POF=0.5,Constraits=1):
        self.__Penalty_param=0
        self.__Pareto=0
        self.__Constraits=Constraits
        self.__lower=0
        self.__upper=0


    def calc_TH(self,X,factor=1):
        return np.pi/2*X**factor
    
     
    def set_Pareto(self, Pareto):
        self.__Pareto=Pareto

     
    def get_Pareto(self):
        return self.__Pareto
    

    def set_Constraits(self, Constraits):
        self.__Constraits=Constraits

     
    def get_Constraits(self):
        return self.__Constraits
    
    
    def set_Penalty_param(self, Penalty_param):
        self.__Penalty_param=Penalty_param

    
    def get_Penalty_param(self):
        return self.__Penalty_param
    

    def set_lower(self,lower):
        self.__lower=lower


    def get_lower(self):
        return self.__lower
    

    def set_upper(self,upper):
        self.__upper=upper

    
    def get_upper(self):
        return self.__upper

    
    def constraints(self,F):
        return self.constraits_1(F,self.get_Penalty_param()) if self.get_Constraits() == 1 else self.constraits_05(F,self.get_Penalty_param())
    
    
    def get_Point_in_G(self):
        return self.__Point_in_G 
    

    def set_Point(self):
        self.__Point_in_G=np.array([*np.random.random((self.CACHE.get_BENCH_CI().get_P(),self.CACHE.get_BENCH_CI().get_Nvar()))*1.0])
        

    def set_POF(self,POF):
        N_Bench = self.CACHE.get_BENCH_CI().get_BENCH_Nvar()
        if N_Bench <=7:
            self.__Point_in_G[:,self.CACHE.get_BENCH_CI().get_M()-1:self.CACHE.get_BENCH_CI().get_Nvar()]=POF
            self.__POF=POF
        elif N_Bench >= 10 and  N_Bench < 14:
            self.__Point_in_G[:,self.CACHE.get_BENCH_CI().get_D()-1:self.CACHE.get_BENCH_CI().get_Nvar()]=POF
            self.__POF=POF
        elif N_Bench == 14:
            self.__Point_in_G[:,self.CACHE.get_BENCH_CI().get_M()-1:self.CACHE.get_BENCH_CI().get_Nvar()]=POF
            self.__POF=POF

    
    def get_POF(self):
        return self.__POF
              
     
    
  
         
    
           
            
        
        

    
       

   


 
    
    





   
 




    
    
    
        
        

    
        
        





