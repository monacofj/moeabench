# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from MoeaBench.core.base_moea import BaseMoea
from MoeaBench.progress import get_active_pbar
import random
from deap import base, creator, tools
import array
import numpy as np

class NSGA2deap(BaseMoea):
    """
    Implementation of the NSGA-II algorithm using the DEAP library.
    
    This class adapts DEAP's NSGA-II to the MoeaBench interface, 
    returning generational history directly.
    """

    def __init__(self, problem=None, population=160, generations=300, seed=1):
        """
        Initializes NSGA2deap.
        
        Args:
            problem: The benchmark problem or experiment.
            population (int): Population size.
            generations (int): Number of generations.
            seed (int): Random seed.
        """
        super().__init__(problem, population, generations, seed)  
        self.toolbox = None

    def _setup_deap(self):
        """Configures the DEAP toolbox and creator."""
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * self.get_M())
        if not hasattr(creator, "Individual"):
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)   
        
        if self.get_population() % 4 != 0:
            raise ValueError(
                f"NSGA2deap error: Population size ({self.get_population()}) must be a multiple of 4 "
                "to satisfy the requirements of the selTournamentDCD operator."
            )

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", self.uniform, 0, 1, self.get_N())
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_float)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_ind)
        
        random.seed(self.seed)
        self.toolbox.decorate("evaluate", tools.DeltaPenality(self._feasible_ind, 1000))
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1/self.get_N())
        self.toolbox.register("select", tools.selNSGA2)

    def uniform(self, low, up, size=None):
        """Generates uniform random values for individuals."""
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low]*size, [up]*size)]

    def _evaluate_ind(self, ind):
        """Evaluates a single DEAP individual."""
        res = self.evaluation_benchmark(np.array(ind))
        return res['F'][0]

    def _feasible_ind(self, ind):
        """Checks feasibility of an individual if constraints exist."""
        res = self.evaluation_benchmark(np.array(ind))
        if 'G' in res:
            return not res.get("feasible", False) 
        return True

    def evaluation(self):
        """
        Executes the NSGA-II optimization.

        Returns:
            tuple: (F_gens, X_gens, F_final, hist_F_nd, hist_X_nd, hist_F_dom, hist_X_dom)
                   where each is a list of arrays (one per generation).
        """
        self._setup_deap()
        pop = self.toolbox.population(n=self.get_population())
        
        # Initial evaluation
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        F_gen_all = []
        X_gen_all = []
        hist_F_non_dominate = []
        hist_X_non_dominate = []
        hist_F_dominate = []
        hist_X_dominate = []
        
        def capture_stats(p):
            F = np.array([ind.fitness.values for ind in p])
            X = np.array([np.array(ind) for ind in p])
            F_gen_all.append(F)
            X_gen_all.append(X)
            
            # Non-dominated and dominated sets
            nd = tools.sortNondominated(p, len(p), first_front_only=True)[0]
            nd_set = set(id(ind) for ind in nd)
            dom = [ind for ind in p if id(ind) not in nd_set]
            
            hist_F_non_dominate.append(np.array([ind.fitness.values for ind in nd]))
            hist_X_non_dominate.append(np.array([np.array(ind) for ind in nd]))
            
            # Handling empty dominated set
            if dom:
                hist_F_dominate.append(np.array([ind.fitness.values for ind in dom]))
                hist_X_dominate.append(np.array([np.array(ind) for ind in dom]))
            else:
                hist_F_dominate.append(np.zeros((0, self.get_M())))
                hist_X_dominate.append(np.zeros((0, self.get_N())))

        capture_stats(pop)
        
        pop = self.toolbox.select(pop, len(pop))
        
        for gen in range(1, self.get_generations() + 1):
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= 0.9:
                    self.toolbox.mate(ind1, ind2)
                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop = self.toolbox.select(pop + offspring, len(pop))
            capture_stats(pop)

            # Update Progress Bar
            pbar = get_active_pbar()
            if pbar:
                pbar.update_to(gen)

            # Custom Stop Criteria
            # We expose the current population for the check
            self.pop = pop 
            self.n_gen = gen # Expose current generation
            if callable(self.stop) and self.stop(self):
                break

        F_final = np.array([ind.fitness.values for ind in pop])
        
        # Store history on self for external access
        self.F_gen_all = F_gen_all
        self.X_gen_all = X_gen_all

        return F_gen_all, X_gen_all, F_final, \
               hist_F_non_dominate, hist_X_non_dominate, \
               hist_F_dominate, hist_X_dominate
