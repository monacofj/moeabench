from pymoo.algorithms.moo.rvea import RVEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from .base_pymoo import BasePymoo

class RVEA_pymoo(BasePymoo):
    """
    Wrapper for Pymoo's RVEA algorithm.
    """
    def evaluation(self):
        """Standard MoeaBench evaluation entry point."""
        ref_dirs = get_reference_directions("energy", self.M, self.population, seed=self.seed)
        mutation = PolynomialMutation(prob=1/self.Nvar, eta=20)
        crossover = SBX(prob=1.0, eta=15)
        
        algorithm = RVEA(ref_dirs, pop_size=self.population, 
                         crossover=crossover, mutation=mutation, **self.kwargs)
        
        return self.run_minimize(algorithm)

      