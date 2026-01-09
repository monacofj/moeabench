from abc import ABC, abstractmethod

class I_metric(ABC):

    @abstractmethod 
    def __init__(self):
        pass
      
      
    @abstractmethod 
    def __call__(self):
        pass
           
    
    @abstractmethod 
    def trace(self):
        pass
              
    
    @abstractmethod 
    def timeplot(self):
        pass
    
   