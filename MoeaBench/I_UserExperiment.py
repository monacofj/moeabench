from abc import ABC, abstractmethod


class I_UserExperiment(ABC):

    @abstractmethod  
    def run(self):
        pass
        

    @abstractmethod
    def load(self):
        pass


    @abstractmethod
    def save(self):
        pass
