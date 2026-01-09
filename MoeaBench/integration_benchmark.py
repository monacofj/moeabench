from abc import ABC, abstractmethod


class integration_benchmark(ABC):
     
     @abstractmethod
     def __init__(self, module_benchmark: object = None, type : str = None, M : int = 3, P : int = 700, K : int = 10, D : int = 2, N : int = 0, n_ieq_constr : int = 1):      
          self.type=type
          self.M = M
          self.P = P
          self.K = K
          self.D = D
          self.module_benchmark = module_benchmark


     def execute(self):
          self._benchmark = self._benchmark()
          return self._benchmark


     def __call__(self, benchmark):
          self._benchmark = benchmark.repository(self.module_benchmark(self.get_type(), self.get_M(), self.get_P(), self.get_K()))
          return self.execute()
     

     def get_type(self):
          return self.type


     def get_M(self):
          return self.M
     

     def get_P(self):
          return self.P
          
          
     def get_K(self):
          return self.K
          

     def get_D(self):
          return self.D
     



    