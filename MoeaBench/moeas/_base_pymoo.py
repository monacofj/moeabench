# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
import numpy as np
from MoeaBench.progress import get_active_pbar
from MoeaBench.core.base_moea import BaseMoea

class PymooHistoryCallback(Callback):
    """
    Callback for Pymoo algorithms to capture generational history and handle stop conditions.
    """
    def __init__(self, stop_func, experiment):
        super().__init__()
        self.stop_func = stop_func
        self.experiment = experiment
        self.hist_f_pop = []
        self.hist_x_pop = []
        self.hist_f_nd = []
        self.hist_x_nd = []
        self.hist_f_dom = []
        self.hist_x_dom = []

    def _record_generation(self, algorithm):
        """Records the current population state."""
        if algorithm.pop is None:
            return
        
        f_pop = algorithm.pop.get("F")
        x_pop = algorithm.pop.get("X")
        f_nd = algorithm.opt.get("F")
        x_nd = algorithm.opt.get("X")

        # Calculate dominated sets
        nd_set_f = set(map(tuple, f_nd))
        ref_dom = [tuple(f) not in nd_set_f for f in f_pop]
        f_dom = f_pop[ref_dom]
        x_dom = x_pop[ref_dom]

        self.hist_f_pop.append(f_pop)
        self.hist_x_pop.append(x_pop)
        self.hist_f_nd.append(f_nd)
        self.hist_x_nd.append(x_nd)
        self.hist_f_dom.append(f_dom)
        self.hist_x_dom.append(x_dom)

    def notify(self, algorithm):
        """Captures population data and checks for termination."""
        self._record_generation(algorithm)

        # Progress bar and stop condition
        pbar = get_active_pbar()
        if pbar:
            progress = algorithm.n_gen
            if callable(self.stop_func):
                # Pass the algorithm instance as context
                stop_val = self.stop_func(algorithm)
                if isinstance(stop_val, bool) and stop_val:
                    progress = 1.0
                    algorithm.termination.force_termination = True
                elif isinstance(stop_val, (int, float)):
                    progress = stop_val
                    if stop_val >= 1.0:
                        algorithm.termination.force_termination = True
            pbar.update_to(progress)
        elif callable(self.stop_func) and self.stop_func(algorithm):
            algorithm.termination.force_termination = True

class BasePymoo(BaseMoea, Problem):
    """
    Base class for all Pymoo-based MOEAs in MoeaBench.
    """
    def __init__(self, problem, population=None, generations=None, seed=None, stop=None, **kwargs):
        # BaseMoea init
        BaseMoea.__init__(self, problem, population, generations, seed)
        
        # Determine Nvar, M, n_ieq from problem
        self.Nvar = self.get_N()
        self.M = self.get_M()
        self.n_ieq = self.get_n_ieq_constr()
        self.stop = stop
        self.kwargs = kwargs # Store flexible parameters
        
        # Pymoo Problem init
        Problem.__init__(self, n_var=self.Nvar, n_obj=self.M, n_ieq_constr=self.n_ieq, 
                         xl=np.zeros(self.Nvar), xu=np.ones(self.Nvar))

    def _evaluate(self, x, out, *args, **kwargs):
        """Pymoo evaluation interface."""
        result = self.get_problem().evaluation(x, self.n_ieq)
        out["F"] = result['F']
        if "G" in result:
            out["G"] = result['G']

    def run_minimize(self, algorithm_instance):
        """
        Helper to run the Pymoo minimize process with standard callback.
        """
        callback = PymooHistoryCallback(self.stop, self.problem)
        
        res = minimize(
            self,
            algorithm_instance,
            termination=('n_gen', self.generations),
            seed=self.seed,
            save_history=False,
            verbose=False,
            callback=callback
        )
        
        # Store history on self for external access (e.g., snapshots)
        self.F_gen_all = callback.hist_f_pop
        self.X_gen_all = callback.hist_x_pop
        
        return callback.hist_f_pop, callback.hist_x_pop, res.F, \
               callback.hist_f_nd, callback.hist_x_nd, \
               callback.hist_f_dom, callback.hist_x_dom
