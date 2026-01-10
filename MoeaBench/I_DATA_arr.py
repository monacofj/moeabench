# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
        


 
