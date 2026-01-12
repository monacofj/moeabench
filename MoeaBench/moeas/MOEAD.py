from .base_moea_wrapper import BaseMoeaWrapper
from .kernel_moea.MOEAD_pymoo import MOEAD_pymoo

class MOEAD(BaseMoeaWrapper):
    """
    Multi-objective Evolutionary Algorithm based on Decomposition (MOEA/D).
    
    Decomposes a multi-objective optimization problem into a number of scalar 
    optimization subproblems and optimizes them simultaneously.
    
    References:
        Zhang & Li (2007). MOEA/D: A Multiobjective Evolutionary Algorithm 
        Based on Decomposition. IEEE Trans. Evol. Comput.
    """
    def __init__(self, population=150, generations=300, seed=1):
        super().__init__(MOEAD_pymoo, population, generations, seed)