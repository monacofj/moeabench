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

class MOEAD_pymoo(BasePymoo):
    """
    Wrapper for Pymoo's MOEA/D algorithm.
    """
    def evaluation(self):
        """Standard MoeaBench evaluation entry point."""
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
       
    
