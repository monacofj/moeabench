from abc import ABC, abstractmethod
from  MoeaBench.CACHE import CACHE
import numpy as np


class BaseMoea(ABC):
     
     @abstractmethod
     def __init__(self, problem, population : int = 160, generations :int = 300, seed: int = 1):
          self.__problem=self.update_benchmark(problem)
          self.population=population
          self.generations=generations
          self.__CACHE = CACHE()
          self.seed = seed
     

     def update_benchmark(self,experiment):
          problem  = experiment.benchmark
          samples = problem.POFsamples()
          problem.get_CACHE().DATA_store(problem.__class__.__name__,problem.get_type(),problem.get_M(),problem.get_N(),problem.get_n_ieq_constr(),samples,problem.get_P() ,problem.get_K()) 
          return problem
     

     @abstractmethod
     def evaluation(self):
          pass


     @property
     def generations(self):
          return self._generations
     

     @generations.setter
     def generations(self,value):
          self._generations = value


     @property
     def population(self):
         return self._population
        

     @population.setter
     def population(self, value):
          self._population = value
       
       
     def get_CACHE(self):
          return self.__CACHE
     

     def get_generations(self):
          return self.generations
     

     def get_population(self):
          return self.population


     def get_problem(self):
          return self.__problem
     

     def get_M(self):
          return self.get_problem().get_CACHE().get_BENCH_CI().get_M()
     

     def get_N(self):
          return self.get_problem().get_CACHE().get_BENCH_CI().get_Nvar()
     

     def get_n_ieq_constr(self):
          return self.get_problem().get_CACHE().get_BENCH_CI().get_n_ieq_constr()
     

     def evaluation_benchmark(self,X):
          return self.get_problem().evaluation(np.array([X]),self.get_n_ieq_constr())
     

  

     


     


