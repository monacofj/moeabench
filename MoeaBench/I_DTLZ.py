from abc import ABC, abstractmethod

class I_DTLZ(ABC):


    @abstractmethod
    def F1(self):
        pass


    @abstractmethod
    def F2(self):
        pass


    @abstractmethod
    def F3(sel):
        pass


    @abstractmethod
    def Fm1(self):
        pass


    @abstractmethod
    def Fm(self):
        pass


    @abstractmethod
    def FiFj(self):
        pass


    @abstractmethod
    def Gj(self):
        pass


    @abstractmethod
    def Gm(self):
        pass


    @abstractmethod
    def calc_F_M(self):
        pass


    @abstractmethod
    def calc_MinFiFj(self):
        pass


    @abstractmethod
    def calc_H(self):
        pass
    

    @abstractmethod
    def param_F(self):
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
    def evaluation(self):
        pass




