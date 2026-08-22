# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .base_moea_wrapper import BaseMoeaWrapper
from ._unsga_pymoo import UNSGA_pymoo

class U_NSGA3(BaseMoeaWrapper):
    """
    Unified Non-dominated Sorting Genetic Algorithm III (U-NSGA-III).
    
    An extension of NSGA-III that aims to handle both single- and 
    multi-objective optimization problems in a unified manner.

    Keyword Args:
        ref_dirs_seed (int, optional): Fixed seed for the energy reference
            directions. If omitted or ``None``, a distinct deterministic seed
            is derived from each run seed. This wrapper-specific argument is
            consumed by MoeaBench and is not forwarded to pymoo.
    
    References:
        Seada & Deb (2015). U-NSGA-III: A Unified Evolutionary Optimization 
        Procedure for Both Single and Multi-objective Optimization. 
        Kalyanpur, India.
    """
    def __init__(self, population=None, generations=None, seed=None, **kwargs):
        super().__init__(UNSGA_pymoo, population, generations, seed, **kwargs)
