# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod

class I_BENCH_conf(ABC):
    
    @abstractmethod
    def get_M(self):
        pass
         

    @abstractmethod
    def set_M(self):
        pass


    @abstractmethod
    def set_K(self):
        pass


    @abstractmethod
    def get_K(self):
        pass

    
    @abstractmethod
    def get_Nvar(self):
        pass
    

    @abstractmethod
    def set_Nvar(self):
        pass
        

    @abstractmethod
    def get_D(self):
        pass
    

    @abstractmethod
    def set_D(self):
        pass
     

    @abstractmethod
    def get_BENCH(self):
        pass


    @abstractmethod
    def set_BENCH(self):
        pass
    

    @abstractmethod
    def get_BENCH_Nvar(self):
        pass


    @abstractmethod
    def get_n_ieq_constr(self):
        pass
       

    @abstractmethod
    def get_P(self):
        pass


    @abstractmethod  
    def set_FILE(self):
        pass
    
    
    @abstractmethod
    def get_FILE(self):
        pass


    @abstractmethod
    def set(self):
        pass


    @abstractmethod
    def set_user(self):
        pass








    



