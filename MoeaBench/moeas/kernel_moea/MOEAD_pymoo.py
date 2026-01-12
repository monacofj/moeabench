from pymoo.algorithms.moo.moead import MOEAD
from pymoo.decomposition.pbi import PBI
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from .base_pymoo import BasePymoo

class MOEAD_pymoo(BasePymoo):
    """
    Wrapper for Pymoo's MOEA/D algorithm.
    """
    def evaluation(self):
        """Standard MoeaBench evaluation entry point."""
        ref_dirs = get_reference_directions("energy", self.M, self.population, seed=self.seed)
        mutation = PolynomialMutation(prob=1/self.Nvar, eta=20)
        crossover = SBX(prob=1.0, eta=15)
        
        algorithm = MOEAD(ref_dirs, crossover=crossover, mutation=mutation, 
                          decomposition=PBI(eps=0.0, theta=5), **self.kwargs)
        
        return self.run_minimize(algorithm)
       
    
