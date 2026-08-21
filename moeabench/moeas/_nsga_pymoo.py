# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from moeabench.progress import get_active_pbar


from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from ._base_pymoo import BasePymoo
from moeabench.core.seeding import UINT32_MAX, derive_component_seed, validate_seed


REFERENCE_DIRECTIONS_COMPONENT = "nsga3.reference_directions"

class NSGA_pymoo(BasePymoo):
    """
    Wrapper for Pymoo's NSGA-III algorithm.
    """
    def evaluation(self):
        """Standard moeabench evaluation entry point."""
        algorithm_kwargs = dict(self.kwargs)
        requested_seed = algorithm_kwargs.pop("ref_dirs_seed", None)
        if requested_seed is None:
            effective_seed = derive_component_seed(
                self.seed, REFERENCE_DIRECTIONS_COMPONENT
            )
        else:
            effective_seed = validate_seed(
                requested_seed, name="ref_dirs_seed", max_value=UINT32_MAX
            )

        self.component_seeds = {REFERENCE_DIRECTIONS_COMPONENT: effective_seed}
        ref_dirs = get_reference_directions(
            "energy", self.M, self.population, seed=effective_seed
        )
        mutation = PolynomialMutation(prob=1/self.Nvar, eta=20)
        crossover = SBX(prob=1.0, eta=15)
        
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=self.population, 
                          crossover=crossover, mutation=mutation, **algorithm_kwargs)
        
        return self.run_minimize(algorithm)

    


    
