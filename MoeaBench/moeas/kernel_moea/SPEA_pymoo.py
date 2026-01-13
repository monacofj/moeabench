# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from .base_pymoo import BasePymoo

class SPEA_pymoo(BasePymoo):
    """
    Wrapper for Pymoo's SPEA2 algorithm.
    """
    def evaluation(self):
        """Standard MoeaBench evaluation entry point."""
        ref_dirs = get_reference_directions("energy", self.M, self.population, seed=self.seed)
        mutation = PolynomialMutation(prob=1/self.Nvar, eta=20)
        crossover = SBX(prob=1.0, eta=15)
        
        algorithm = SPEA2(ref_dirs=ref_dirs, pop_size=self.population, 
                          crossover=crossover, mutation=mutation, **self.kwargs)
        
        return self.run_minimize(algorithm)
        


    

