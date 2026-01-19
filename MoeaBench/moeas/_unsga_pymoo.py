# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from ._base_pymoo import BasePymoo

class UNSGA_pymoo(BasePymoo):
    """
    Wrapper for Pymoo's U-NSGA-III algorithm.
    """
    def evaluation(self):
        """Standard MoeaBench evaluation entry point."""
        mutation = PolynomialMutation(prob=1/self.Nvar, eta=20)
        crossover = SBX(prob=1.0, eta=15)
        
        algorithm = UNSGA3(pop_size=self.population, 
                           crossover=crossover, mutation=mutation)
        
        return self.run_minimize(algorithm)
