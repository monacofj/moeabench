from abc import ABC, abstractmethod

class I_memory(ABC):
   
    @abstractmethod
    def get_DATA_conf(self):
          pass
  
    
    @abstractmethod
    def edit_DATA_conf(self):
          pass
       
    
    @abstractmethod
    def get_BENCH_conf(self):
          pass
    
    
    @abstractmethod
    def set_BENCH_CI(self):
          pass

    
    @abstractmethod
    def get_BENCH_CI(self):
        pass
    

    @abstractmethod
    def DATA_store(self):
         pass