from abc import ABC, abstractmethod

class I_constraints(ABC):


    @abstractmethod
    def eval_cons(self):
        pass


    @abstractmethod
    def constraits_1(self):
        pass


    @abstractmethod
    def constraits_05(self):
        pass
