# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod


class I_UserMoeaBench(ABC):
    
    @abstractmethod
    def spaceplot(self):
        pass


    @abstractmethod
    def surfaceplot(self):
        pass
        
  
    @abstractmethod
    def add_benchmark(self):
        pass


    @abstractmethod
    def add_moea(self):
        pass


    
