# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod

class I_IBENCH(ABC):
    
    @abstractmethod
    def set_Pareto(self):
        pass


    @abstractmethod 
    def get_Pareto(self):
        pass
       
    
    @abstractmethod
    def set_Constraits(self):
        pass
        

    @abstractmethod
    def get_Constraits(self):
        pass
       
    
    @abstractmethod
    def set_Penalty_param(self):
        pass
      

    @abstractmethod
    def get_Penalty_param(self):
        pass


    @abstractmethod
    def set_BENCH_conf(self):
        pass

    
    @abstractmethod
    def calc_TH(self):
        pass


    @abstractmethod
    def constraints(self):
        pass

    
    @abstractmethod
    def set_lower(self):
        pass


    @abstractmethod
    def get_lower(self):
        pass


    @abstractmethod
    def set_upper(self):
        pass


    @abstractmethod
    def get_upper(self):
        pass


    @abstractmethod
    def get_CACHE(self):
        pass




    




    




        






    

    


    


    


