from abc import ABC, abstractmethod

class I_DATA_conf(ABC):
    

    @abstractmethod
    def get_arr_DATA(self):
        pass


    @abstractmethod
    def get_description(self):
        pass


    @abstractmethod
    def get_generations(self):
        pass


    @abstractmethod
    def get_population(self):
        pass


    @abstractmethod
    def get_problem(self):
        pass

    
    @abstractmethod
    def set_DATA_MOEA(self):
        pass


    @abstractmethod
    def get_DATA_MOEA(self):
        pass


    @abstractmethod
    def get_F_GEN(self):
        pass


    @abstractmethod
    def get_F_gen_non_dominate(self):
        pass


    @abstractmethod
    def get_F_gen_dominate(self):
        pass
    

    @abstractmethod
    def get_X_GEN(self):
        pass

    
    @abstractmethod
    def get_X_gen_non_dominate(self):
        pass
      
    
    @abstractmethod
    def get_X_gen_dominate(self):
        pass


    

    



    
  

   

    
   