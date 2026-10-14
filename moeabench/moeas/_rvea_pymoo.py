# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.algorithms.moo.rvea import RVEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from ._base_pymoo import BasePymoo
from ._operators_pymoo import standard_polynomial_mutation

class RVEA_pymoo(BasePymoo):
    """
    Wrapper for Pymoo's RVEA algorithm.
    """
    def evaluation(self):
        """Standard moeabench evaluation entry point."""
        ref_dirs = get_reference_directions("energy", self.M, self.population, seed=self.seed)
        mutation = standard_polynomial_mutation(self.Nvar)
        crossover = SBX(prob=1.0, eta=15)
        
        algorithm = RVEA(ref_dirs, pop_size=self.population, 
                         crossover=crossover, mutation=mutation, **self.kwargs)
        
        return self.run_minimize(algorithm)

      