# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
