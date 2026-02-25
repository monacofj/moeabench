# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""     
        - Benchmarks:       
            MoeaBench uses several evolutionary algorithms from the pymoo library.
               ...
               - NSGA-III:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.NSGA_III(args) 
                      - [NSGA-III](https://moeabench-rgb.github.io/MoeaBench/algorithms/NSGA3/) information about the genetic algorithm
               ... 
               - U-NSGA-III:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.U_NSGA_III(args)
                      - [U-NSGA-3](https://moeabench-rgb.github.io/MoeaBench/algorithms/UNSGA3/) information about the genetic algorithm
                ...
               - SPEA-II:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.SPEA_II(args) 
                      - [SPEA-II](https://moeabench-rgb.github.io/MoeaBench/algorithms/SPEA2/) information about the genetic algorithm
               ... 
               - MOEA/D:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.MOEAD(args) 
                      - [MOEA/D](https://moeabench-rgb.github.io/MoeaBench/algorithms/MOEAD/) information about the genetic algorithm
                 ... 
               - RVEA:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.RVEA(args) 
                      - [RVEA](https://moeabench-rgb.github.io/MoeaBench/algorithms/RVEA/) information about the genetic algorithm
                 ...       
               - my_new_moea:      
                      - sinxtase:
                      experiment.benchmark = moeabench.moeas.my_new_moea(args)       
                      - [my_new_moea](https://moeabench-rgb.github.io/MoeaBench/implement_moea/memory/memory/) information about the method, 
                 ...          
"""

from .NSGA2deap import NSGA2deap
from .NSGA3 import NSGA3
from .U_NSGA3 import U_NSGA3
from .SPEA2 import SPEA2
from .MOEAD import MOEAD
from .RVEA import RVEA
from ..core.base_moea import BaseMoea

# Optional: Convenience alias for NSGA2
NSGA2 = NSGA2deap
