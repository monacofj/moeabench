# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .base_moea_wrapper import BaseMoeaWrapper
from ._moead_pymoo import MOEAD_pymoo

from .moead_configs import get_moead_params

class MOEAD(BaseMoeaWrapper):
    """
    Multi-objective Evolutionary Algorithm based on Decomposition (MOEA/D).
    
    Decomposes a multi-objective optimization problem into a number of scalar 
    optimization subproblems and optimizes them simultaneously.
    
    References:
        Zhang & Li (2007). MOEA/D: A Multiobjective Evolutionary Algorithm 
        Based on Decomposition. IEEE Trans. Evol. Comput.
    """
    def __init__(self, population=None, generations=None, seed=None, **kwargs):
        super().__init__(MOEAD_pymoo, population, generations, seed, **kwargs)

    def __call__(self, experiment, **kwargs):
        """Overrides base call to inject problem-specific parameters."""
        # Get problem name from experiment (mop object)
        problem_name = experiment.__class__.__name__
        
        # Look up tuned params
        tuned_params = get_moead_params(problem_name)
        
        # Merge: kwargs (explicit user) > tuned_params (registry) > defaults
        # We update self._kwargs so the engine receives them
        for k, v in tuned_params.items():
            if k not in self._kwargs:
                self._kwargs[k] = v
                
        return super().__call__(experiment, **kwargs)