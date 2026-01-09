from abc import ABC, abstractmethod
from MoeaBench.CACHE_bk_user import CACHE_bk_user

class BaseBenchmark(ABC):
     
     @abstractmethod
     def __init__(self, types: str, M: int, P: int, K : int = None, N  : int = None, D : int = None, n_ieq_constr :int =1):
        self.benchmark = self
        self.M=M
        self.P=P
        self.K=K
        self.D=D
        self.__N=self.calc_N(K,M)
        self.__n_ieq_constr=n_ieq_constr
        self.__CACHE=CACHE_bk_user()
        self.__type=types
    

     def get_CACHE(self):
          return self.__CACHE
     

     def get_M(self):
          return self.M
     

     @property
     def M(self):
          return self._M 
     

     @M.setter 
     def M(self,value):
          self._M = value
          if hasattr(self,"_K"):
               self.__N = self.calc_N(self.get_K(),value)


     def get_P(self):
          return self.P
     

     @property
     def P(self):
          return self._P
     

     @P.setter 
     def P(self,value):
          self._P = value


     def get_K(self):
          return self.K
     

     @property
     def K(self):
          return self._K
     

     @K.setter 
     def K(self,value):
          self._K = value
          self.__N = self.calc_N(value,self.get_M())


     def get_N(self):
          return self.__N
     

     def get_D(self):
          return self.D
     

     @property
     def D(self):
          return self._D
     

     @D.setter 
     def D(self,value):
          self._D = value
     

     def get_n_ieq_constr(self):
          return self.__n_ieq_constr
     

     def get_type(self):
          return self.__type


     @abstractmethod
     def evaluation(self):
          pass
     

     @abstractmethod
     def POFsamples(self):
          pass


     @abstractmethod
     def set_Point_in_G(self):
          pass

     
     @abstractmethod
     def get_Point_in_G(self):
          pass

      

     



