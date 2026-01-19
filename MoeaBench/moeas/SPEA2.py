# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .base_moea_wrapper import BaseMoeaWrapper
from ._spea_pymoo import SPEA_pymoo

class SPEA2(BaseMoeaWrapper):
    """
    Strength Pareto Evolutionary Algorithm 2 (SPEA2).
    
    A multi-objective evolutionary algorithm that uses a fine-grained 
    fitness assignment strategy and an enhanced archive truncation 
    method to maintain diversity and elitism.
    
    References:
        Zitzler, Laumanns, & Thiele (2001). SPEA2: Improving the Strength 
        Pareto Evolutionary Algorithm. Technical Report 103, ETH Zurich.
    """
    def __init__(self, population=150, generations=300, seed=1, **kwargs):
        super().__init__(SPEA_pymoo, population, generations, seed, **kwargs)