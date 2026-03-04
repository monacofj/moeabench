# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.decomposition.pbi import PBI
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from ._base_pymoo import BasePymoo

import numpy as np

class MOEAD_pymoo(BasePymoo):
    """
    Wrapper for Pymoo's MOEA/D algorithm.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MOEA/D in pymoo does not support constraints natively.
        # We spoof the problem to appear unconstrained to pymoo,
        # but apply a death penalty inside _evaluate.
        self._og_n_ieq = self.n_ieq_constr
        self.n_ieq_constr = 0

    def _evaluate(self, x, out, *args, **kwargs):
        """Override to apply death penalty for constraints."""
        result = self.get_problem().evaluation(x, self._og_n_ieq)
        F = np.array(result['F'])
        if "G" in result and self._og_n_ieq > 0:
            G = np.array(result['G'])
            CV = np.maximum(0, G).sum(axis=1) # Sum of constraint violations
            F = F + (CV * 1e6)[:, None] # Death penalty applied to all objectives
        out["F"] = F
        # Intentionally omitting out["G"] so pymoo thinks there are no constraints

    def evaluation(self):
        """Standard moeabench evaluation entry point."""
        ref_dirs = get_reference_directions("energy", self.M, self.population, seed=self.seed)
        mutation = PolynomialMutation(prob=1/self.Nvar, eta=20)
        crossover = SBX(prob=1.0, eta=15)
        
        # Extract MOEAD params from kwargs if present, else use v0.7.6 defaults
        decomposition = self.kwargs.pop('decomposition', PBI(eps=0.0, theta=5))
        n_neighbors = self.kwargs.pop('n_neighbors', 15)
        prob_neighbor_mating = self.kwargs.pop('prob_neighbor_mating', 0.9)
        

        algorithm = MOEAD(ref_dirs, 
                          crossover=crossover, 
                          mutation=mutation, 
                          decomposition=decomposition,
                          n_neighbors=n_neighbors,
                          prob_neighbor_mating=prob_neighbor_mating,
                          **self.kwargs)
        
        return self.run_minimize(algorithm)
       
    
