# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""     
MoeaBench Benchmark Problems.

References:
    [DTLZ] K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, "Scalable Multi-Objective 
           Optimization Test Problems," Proc. IEEE CEC, 2002.
    [DPF]  L. Zhen, M. Li, R. Cheng, D. Peng, and X. Yao, "Multiobjective Test Problems 
           with Degenerate Pareto Fronts," IEEE Trans. Evol. Comput., 2018.
"""

# Modernized Benchmarks
from .base_mop import BaseMop

from .DTLZ1 import DTLZ1
from .DTLZ2 import DTLZ2
from .DTLZ3 import DTLZ3
from .DTLZ4 import DTLZ4
from .DTLZ5 import DTLZ5
from .DTLZ6 import DTLZ6
from .DTLZ7 import DTLZ7
from .DTLZ8 import DTLZ8
from .DTLZ9 import DTLZ9

from .base_dpf import BaseDPF
from .DPF1 import DPF1
from .DPF2 import DPF2
from .DPF3 import DPF3
from .DPF4 import DPF4
from .DPF5 import DPF5

# Legacy Compatibility Adapters - Removed as CACHE is deleted
# from MoeaBench.CACHE import CACHE
# from MoeaBench.CACHE_bk_user import CACHE_bk_user

# Enum for problems
from .E_problems import E_problems
from .E_problems_bk import E_problems_bk

# Legacy dynamic loading for user plugins could be re-added here if strictly necessary,
# but for core benchmarks we prefer explicit imports.