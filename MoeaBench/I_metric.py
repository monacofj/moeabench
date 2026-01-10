# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod

class I_metric(ABC):

    @abstractmethod 
    def __init__(self):
        pass
      
      
    @abstractmethod 
    def __call__(self):
        pass
           
    
    @abstractmethod 
    def trace(self):
        pass
              
    
    @abstractmethod 
    def timeplot(self):
        pass
    
   