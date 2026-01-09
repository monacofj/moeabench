from abc import ABC, abstractmethod


class I_DATA_arr(ABC):
   
    @abstractmethod
    def get_elements(self):
        pass
    
    
    @abstractmethod
    def add_T(self):
        pass


    @abstractmethod
    def clear(self):
        pass
        


 
