# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .base_moea_wrapper import BaseMoeaWrapper
from ._nsga_pymoo import NSGA_pymoo

class NSGA3(BaseMoeaWrapper):
    """
    Non-dominated Sorting Genetic Algorithm III (NSGA-III).
    
    A many-objective evolutionary algorithm that uses reference directions
    to maintain diversity in high-dimensional objective spaces.
    
    References:
        Deb & Jain (2014). An Evolutionary Many-Objective Optimization Algorithm 
        Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving 
        Problems With Box Constraints. IEEE Trans. Evol. Comput.
    """
    def __init__(self, population=None, generations=None, seed=None, **kwargs):
        super().__init__(NSGA_pymoo, population, generations, seed, **kwargs)