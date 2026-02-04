# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional, Tuple, List, Union

class BaseMoea(ABC):
    """
    Abstract Base Class for all Multi-Objective Evolutionary Algorithms (MOEAs).
    
    This class defines the interface and common attributes for algorithms that 
    optimize multiple objectives on a benchmark problem.
    """
     
    def __init__(self, problem: Any, population: Optional[int] = None, generations: Optional[int] = None, seed: Optional[int] = None) -> None:
        """
        Initializes the MOEA with a problem and parameters.

        Args:
            problem: The benchmark problem to optimize (Experiment or Benchmark object).
            population (int): The number of individuals in each generation.
            generations (int): The total number of generations to run.
            seed (int): Random seed for reproducibility.
        """
        self.problem = problem
        self.population = population
        self.generations = generations
        self.seed = seed
        self.stop = None

    @abstractmethod
    def evaluation(self) -> Any:
        """
        Executes the optimization process.
        
        Returns:
            A history of populations, or a structured result object depending on implementation.
            Phase 1 target: return (F_gens, X_gens, F_final, F_nd_history, X_nd_history, F_dom_history, X_dom_history)
        """
        pass

    @property
    def generations(self) -> int:
        """int: The number of generations to run."""
        return self._generations
     
    @generations.setter
    def generations(self, value: int) -> None:
        self._generations = int(value)

    @property
    def population(self) -> int:
        """int: The size of the population."""
        return self._population
        
    @population.setter
    def population(self, value: int) -> None:
        self._population = int(value)

    def get_generations(self) -> int:
        """Legacy helper for generations."""
        return self.generations
     
    def get_population(self) -> int:
        """Legacy helper for population."""
        return self.population

    def get_problem(self) -> Any:
        """Returns the benchmark problem instance."""
        # Typically the experiment object passed in __init__ has a .benchmark attribute
        if hasattr(self.problem, 'benchmark'):
            return self.problem.benchmark
        return self.problem
     
    def get_M(self) -> int:
        """Returns the number of objectives of the problem."""
        problem = self.get_problem()
        if hasattr(problem, 'M'):
            return problem.M
        if hasattr(problem, 'get_M'):
            return problem.get_M()
        raise AttributeError(f"Problem {problem} has no attribute 'M' or 'get_M()'")
     
    def get_N(self) -> int:
        """Returns the number of decision variables of the problem."""
        problem = self.get_problem()
        if hasattr(problem, 'N'):
            return problem.N
        if hasattr(problem, 'get_Nvar'):
            return problem.get_Nvar()
        # Fallback for DTLZ problems if M and K are present
        if hasattr(problem, 'M') and hasattr(problem, 'K'):
            return problem.M + problem.K - 1
        raise AttributeError(f"Problem {problem} has no attribute 'N' or 'get_Nvar()'")
     
    def get_n_ieq_constr(self) -> int:
        """Returns the number of inequality constraints of the problem."""
        problem = self.get_problem()
        if hasattr(problem, 'n_ieq_constr'):
            return problem.n_ieq_constr
        if hasattr(problem, 'get_n_ieq_constr'):
            return problem.get_n_ieq_constr()
        return 0
     
    def evaluation_benchmark(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluates a set of decision variables against the benchmark problem.

        Args:
            X (np.ndarray): Array of decision variable sets to evaluate.

        Returns:
            dict: Evaluation results containing 'F' (objectives) and optionally 'G' (constraints).
        """
        return self.get_problem().evaluation(np.array([X]), self.get_n_ieq_constr())

    def __getstate__(self) -> Dict[str, Any]:
        """Custom state for pickling to avoid non-picklable attributes."""
        state = self.__dict__.copy()
        # DEAP toolbox is not picklable and should be recreated on demand or after loading
        if 'toolbox' in state:
            state['toolbox'] = None
        return state
