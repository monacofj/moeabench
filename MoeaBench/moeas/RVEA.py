# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .base_moea_wrapper import BaseMoeaWrapper
from .kernel_moea.RVEA_pymoo import RVEA_pymoo

class RVEA(BaseMoeaWrapper):
    """
    Reference Vector Guided Evolutionary Algorithm (RVEA).
    
    A many-objective evolutionary algorithm that uses reference vectors 
    to balance convergence and diversity.
    
    References:
        Cheng, Jin, Olhofer, & Sendhoff (2016). A Reference Vector Guided 
        Evolutionary Algorithm for Many-objective Optimization. IEEE Trans. 
        Evol. Comput.
    """
    def __init__(self, population=150, generations=300, seed=1, **kwargs):
        super().__init__(RVEA_pymoo, population, generations, seed, **kwargs)