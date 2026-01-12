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
    def __init__(self, population=150, generations=300, seed=1):
        super().__init__(RVEA_pymoo, population, generations, seed)