from abc import ABC, abstractmethod


class I_UserMoeaBench(ABC):
    
    @abstractmethod
    def spaceplot(self):
        pass


    @abstractmethod
    def surfaceplot(self):
        pass
        
  
    @abstractmethod
    def add_benchmark(self):
        pass


    @abstractmethod
    def add_moea(self):
        pass


    
