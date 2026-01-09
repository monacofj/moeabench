from abc import ABC, abstractmethod

class I_DPF(ABC):


    @abstractmethod
    def B1(self):
        pass


    @abstractmethod
    def B2(self):
        pass


    @abstractmethod
    def Bmd1(sel):
        pass


    @abstractmethod
    def Y1 (self):
        pass


    @abstractmethod
    def Y2 (self):
        pass


    @abstractmethod
    def Yd1 (self):
        pass


    @abstractmethod
    def Yd (self):
        pass

     
    @abstractmethod
    def FD(self):
        pass


    @abstractmethod
    def FD1(self):
        pass
     
     
    @abstractmethod
    def FM(self):
        pass

    
    @abstractmethod
    def param_CHAOS(self):
        pass
    
    
    @abstractmethod
    def calc_F_P(self):
        pass


    @abstractmethod
    def calc_F_D(self):
        pass

    
    @abstractmethod
    def calc_F_C(self):
        pass

    
    @abstractmethod
    def calc_NU_(self):
        pass
    
    
    @abstractmethod
    def calc_NU(self):
        pass
    

    @abstractmethod
    def param_F(self):
        pass
    

    @abstractmethod
    def calc_F_PD(self):
        pass


    @abstractmethod
    def calc_f(self):
        pass


    @abstractmethod
    def calc_g(self):
        pass


    @abstractmethod
    def calc_gijx(self):
        pass
    
    
    @abstractmethod
    def show_in(self):
        pass


    @abstractmethod
    def minimize(self):
        pass

    
    @abstractmethod
    def get_method(self):
        pass


    @abstractmethod
    def get_method_R1(self):
        pass


    @abstractmethod
    def get_method_R2(self):
        pass









