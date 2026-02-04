# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# from .I_UserExperiment import I_UserExperiment # Legacy removed
from .base import Reportable
from .run import Run, SmartArray, Population
import numpy as np
import inspect
import os
import numpy as np
import inspect
import os
from ..progress import get_progress_bar, set_active_pbar
from typing import Optional, List, Union, Any, Iterator, Dict

class JoinedPopulation:
    def __init__(self, pops: List[Population], source: Any = None) -> None:
        self.pops = pops
        self.source = source
        self.label = "Population"
        self.name = getattr(source, 'name', 'Population') if source else 'Population'
        
    @property
    def objectives(self) -> SmartArray:
        if not self.pops:
             return SmartArray(np.array([]), label="Population (Objectives)", axis_label="Objective")
        data = np.vstack([p.objectives for p in self.pops])
        return SmartArray(data, label="Population (Objectives)", axis_label="Objective")

    @property
    def variables(self) -> SmartArray:
        if not self.pops:
             return SmartArray(np.array([]), label="Population (Variables)", axis_label="Variable")
        data = np.vstack([p.variables for p in self.pops])
        return SmartArray(data, label="Population (Variables)", axis_label="Variable")

    @property
    def objs(self) -> SmartArray: return self.objectives

    @property
    def vars(self) -> SmartArray: return self.variables

    def __len__(self) -> int:
        if not self.pops: return 0
        return sum(len(p) for p in self.pops)

class experiment(Reportable):
    def __init__(self, mop: Optional[Any] = None, moea: Optional[Any] = None) -> None:
        self._runs: List[Run] = []
        self._mop: Any = None
        self._moea: Any = None
        self._stop: Any = None
        self._name: str = "experiment"
        self._repeat: int = 1

        
        # Use properties for auto-instantiation and validation
        if mop is not None:
            self.mop = mop
        if moea is not None:
            self.moea = moea
        
        # Internal state for execution
        self.result: Any = None 

    def __iter__(self) -> Iterator[Run]:
        return iter(self._runs)

    def __getitem__(self, index: int) -> Run:
        return self._runs[index]

    def __len__(self) -> int:
        return len(self._runs)

    @property
    def name(self) -> str: return self._name
    @name.setter
    def name(self, value: str) -> None: self._name = value
    
    @property
    def repeat(self) -> int:
        """Returns the default number of repetitions for this experiment."""
        return self._repeat
    @repeat.setter
    def repeat(self, value: int) -> None:
        """Sets the default number of repetitions for this experiment."""
        if not isinstance(value, int) or value < 1:
            raise ValueError("Repeat must be a positive integer.")
        self._repeat = value
    
    @property
    def mop(self) -> Any: return self._mop
    @mop.setter
    def mop(self, value: Any) -> None:
        # Auto-instantiate if it's a factory (callable but not the final instance with get_CACHE)
        # We assume if it's callable and missing core traits, it's a factory.
        if callable(value) and not hasattr(value, 'get_CACHE'):
             try:
                 self._mop = value()
             except Exception as e:
                 # Fallback or error logging
                 print(f"Warning: Could not instantiate MOP factory: {e}")
                 self._mop = value
        else:
            self._mop = value

    @property
    def moea(self) -> Any: return self._moea
    @moea.setter
    def moea(self, value: Any) -> None:
        self._moea = value

    @property
    def runs(self) -> List[Run]:
        """Returns the list of Run objects from this experiment."""
        return self._runs
        # Original code did magic here to instantiate result.
        # We will handle instantiation in run()

    @property
    def stop(self) -> Any: return self._stop
    @stop.setter
    def stop(self, value: Any) -> None: self._stop = value
    
    @property
    def pof(self) -> Any:
        """Legacy compatibility for save mechanism."""
        return self._mop

    # Shortcuts
    @property
    def last_run(self) -> Run:
        if not self._runs: raise IndexError("No runs executed yet")
        return self._runs[-1]

    @property
    def last_pop(self) -> Population:
        return self.last_run.last_pop

    def pop(self, gen: int = -1) -> JoinedPopulation:
        """Aggregate populations from all runs."""
        # This is tricky. API.py says:
        # exp.pop(100).objectives # Objs of 100th gen, or all run.
        # This implies returning a "MultiPopulation" or just all of them?
        # If I return a list of Populations, .objectives won't work on the list.
        # The user's API implies exp.pop(100) returns something that has .objectives.
        # Which likely means CONCATENATED objectives from all runs?
        # "metrics.hypervolume(exp)" handles the list iteration.
        # But "exp.pop().objectives" implies aggregation.
        
        # Let's create a JoinedPopulation
        return JoinedPopulation([run.pop(gen) for run in self._runs], source=self)

    def front(self, gen: int = -1) -> SmartArray:
         """Returns the non-dominated front (objectives) from the aggregate cloud (all runs)."""
         # To get the front of a single run, use: exp.last_run.front()
         return self.superfront(gen)

    def set(self, gen: int = -1) -> SmartArray:
         """Returns the non-dominated decision set (variables) from the aggregate cloud (all runs)."""
         return self.superset(gen)

    def all_fronts(self, gen: int = -1) -> List[SmartArray]:
        """Returns a list of Pareto fronts from all runs."""
        return [run.front(gen) for run in self._runs]

    def all_sets(self, gen: int = -1) -> List[SmartArray]:
        """Returns a list of decision sets from all runs."""
        return [run.set(gen) for run in self._runs]

    def superfront(self, gen: int = -1) -> SmartArray:
        """[Deprecated] Returns the non-dominated front considering all runs combined. Use exp.front() instead."""
        p = self.pop(gen)
        # Create a combined population to apply global filtering
        combined = Population(p.objectives, p.variables, source=self, label="Superfront")
        res = combined.non_dominated().objectives
        if hasattr(res, 'name'): res.name = self.name
        return res

    def superset(self, gen: int = -1) -> SmartArray:
        """[Deprecated] Returns the non-dominated decision set considering all runs combined. Use exp.set() instead."""
        p = self.pop(gen)
        combined = Population(p.objectives, p.variables, source=self, label="Superset")
        res = combined.non_dominated().variables
        if hasattr(res, 'name'): res.name = self.name
        return res

    def non_front(self, gen: int = -1) -> SmartArray:
         """Returns the dominated objectives (non-front) from the aggregate cloud."""
         return self.dominated(gen).objectives

    def non_set(self, gen: int = -1) -> SmartArray:
         """Returns the dominated decision set from the aggregate cloud."""
         return self.dominated(gen).variables

    def dominated(self, gen: int = -1) -> Population:
         """Returns the dominated Population from the aggregate cloud at gen."""
         p = self.pop(gen)
         # Filter global cloud for dominated individuals
         pop = Population(p.objectives, p.variables, source=self, label="Dominated", gen=gen)
         return pop.dominated()

    def non_dominated(self, gen: int = -1) -> Population:
         """Returns the non-dominated Population from the aggregate cloud at gen."""
         p = self.pop(gen)
         # Filter global cloud for non-dominated individuals
         pop = Population(p.objectives, p.variables, source=self, label="Non-dominated", gen=gen)
         return pop.non_dominated()

    def optimal(self, n_points: int = 500) -> Population:
        """Returns a sampling of the true Pareto optimal set and front."""
        if not hasattr(self.mop, 'ps'):
             raise AttributeError(f"MOP {self.mop.__class__.__name__} does not implement ps() sampling.")
        
        # 1. Synchronized Sampling: Get the Pareto Set first
        vars = self.mop.ps(n_points)
        
        # 2. Evaluation: Map the Set to the Front using the problem's own logic
        # This ensures row-by-row correspondence between X and F.
        res = self.mop.evaluation(vars)
        objs = res['F']
        
        # 3. Filtering: Apply non-dominance purely over the theoretical samples
        # This handles cases like DTLZ7 where the condition g=min is not sufficient.
        pop = Population(objs, vars, source=self, label="Optimal")
        pop = pop.non_dominated()
        
        # For analytical optimal fronts, we want the name to be "Optimal" 
        # instead of the experiment name.
        pop.objectives.name = "Optimal"
        pop.variables.name = "Optimal"
        
        return pop

    def optimal_front(self, n_points: int = 500) -> SmartArray:
        """Alias for exp.optimal().objectives"""
        return self.optimal(n_points).objectives

    def optimal_set(self, n_points: int = 500) -> SmartArray:
        """Alias for exp.optimal().variables"""
        return self.optimal(n_points).variables

    # Shortcuts from API.py
    @property
    def objectives(self) -> SmartArray:
        return self.pop().objectives

    @property
    def variables(self) -> SmartArray:
        return self.pop().variables

    # Delegation to mop for MOEA compatibility
    @property
    def M(self) -> int: return self.mop.M
    @property
    def N(self) -> int: return self.mop.N
    
    def get_M(self) -> int: return self.mop.get_M()
    def get_Nvar(self) -> Optional[int]: return self.mop.get_Nvar()
    def get_n_ieq_constr(self) -> int: return self.mop.get_n_ieq_constr()

    def evaluation(self, X: np.ndarray, n_ieq_constr: int = 0) -> Dict[str, np.ndarray]:
        """Delegates evaluation to the internal MOP."""
        return self.mop.evaluation(X, n_ieq_constr)
         
    # Reporting
    def report(self, **kwargs) -> str:
        """Narrative report of the experiment's configuration and status."""
        mop_name = self.mop.__class__.__name__ if self.mop else "None"
        moea_name = self.moea.__class__.__name__ if self.moea else "None"
        
        m_objs = getattr(self.mop, 'M', '?') if self.mop else '?'
        n_vars = getattr(self.mop, 'N', '?') if self.mop else '?'
        
        n_runs = len(self._runs)
        status = "[OK] Populated" if n_runs > 0 else "[EMPTY] Empty (Not run)"
        
        lines = [
            f"--- Experiment Report: {self.name} ---",
            f"  Status: {status}",
            f"  Problem (MOP): {mop_name} (Objectives={m_objs}, Variables={n_vars})",
            f"  Algorithm (MOEA): {moea_name}",
            f"  Configuration: {n_runs}/{self.repeat} runs completed"
        ]
        
        if n_runs > 0:
            last_gen = len(self._runs[0])
            lines.append(f"  Timeline: ~{last_gen} generations per run")
            
        return "\n".join(lines)

    # Execution
    def run(self, repeat: Optional[int] = None, workers: Optional[int] = None, 
            diagnose: bool = False, **kwargs) -> None:
        """
        Executes the optimization experiment for one or more runs.

        Args:
            repeat (int): Number of independent runs to perform. Defaults to self.repeat.
            workers (int): [DEPRECATED] Parallel execution is no longer supported. 
                           All runs are performed serially for stability.
            diagnose (bool): If True, performs automated algorithmic pathology analysis 
                             after execution and prints the rationale. Defaults to False.
            **kwargs: Parameters to override in the MOEA (e.g., generations, population).
        """
        if repeat is not None:
            self.repeat = repeat
        else:
            repeat = self.repeat
        if repeat < 1: repeat = 1
        
        # Propagate overrides to the MOEA
        for key, val in kwargs.items():
            if hasattr(self.moea, key):
                setattr(self.moea, key, val)
        
        # Determine base seed
        base_seed = getattr(self.moea, 'seed', None)
        if base_seed is None:
            base_seed = np.random.randint(0, 1000000)
            if hasattr(self.moea, 'seed'):
                self.moea.seed = base_seed

        # Stop criteria handling
        stop_criteria = kwargs.get('stop', self.stop)
        if hasattr(self.moea, 'stop') and (stop_criteria is not None or 'stop' in kwargs):
            self.moea.stop = stop_criteria

        # Execute serially
        self._run_serial(repeat, base_seed)

        if diagnose and self._runs:
            from .. import diagnostics, metrics
            print("\n--- [Diagnostics] Algorithmic Pathology Report ---")
            
            # Diagnose the last run (most representative of current state)
            run = self.last_run
            try:
                # Calculate quick metrics for diagnosis
                # Note: This assumes MOP has a known front or we use indicators
                diagnosis_metrics = {}
                
                # We need GT for accurate diagnosis
                if hasattr(self.mop, 'pareto_front'):
                    pf = self.mop.pareto_front()
                    if pf is not None:
                         # Use existing metrics module
                         m_igd = metrics.GEN_igd(pf)
                         m_gd = metrics.GEN_gd(pf)
                         
                         pop_f = run.last_pop.objectives
                         diagnosis_metrics['igd'] = m_igd.do(pop_f)
                         diagnosis_metrics['gd'] = m_gd.do(pop_f)
                         
                         # Approximate H_rel if HV available
                         # (Skipping complex HV calculation for 'light' diagnosis unless critical)
                
                # Perform Audit
                result = diagnostics.audit(diagnosis_metrics)
                result.report_show()
                
            except Exception as e:
                print(f"Diagnostics Failed: {e}")
                print("--------------------------------------------------\n")

    def _run_serial(self, repeat: int, base_seed: int) -> None:
        total_gens = getattr(self.moea, 'generations', None)
        outer_pbar = None
        if repeat > 1:
            outer_pbar = get_progress_bar(total=repeat, desc=f"Experiment: {self.name}", position=0)

        for i in range(repeat):
            seed = base_seed + i
            
            inner_pbar = get_progress_bar(total=total_gens, 
                                         desc=f"  Run {i+1}/{repeat}", 
                                         position=1 if repeat > 1 else 0,
                                         leave=False if repeat > 1 else True)
            set_active_pbar(inner_pbar)

            try:
                run_data, _ = self._execute_run(self.moea, self.mop, seed, i+1)
                new_run = Run(run_data, seed, experiment=self, index=i+1)
                self._runs.append(new_run)
            finally:
                inner_pbar.close()
                set_active_pbar(None)
                if outer_pbar:
                    outer_pbar.update_to(i + 1)

        if outer_pbar:
            outer_pbar.close()

    def _execute_run(self, moea, mop, seed, index):
        """Internal helper for executing a single MOEA run."""
        # Inject context
        moea.problem = mop 
        if hasattr(moea, 'seed'):
            moea.seed = seed

        
        try:
            # Execute
            if hasattr(moea, 'evaluation'):
                # New-style API
                data_payload = moea.evaluation()
            else:
                # Legacy Flow
                current_result = moea(mop, None, None, seed)
                raw_result = current_result[0] if isinstance(current_result, tuple) else current_result
                
                if hasattr(raw_result, 'edit_DATA_conf'):
                     moea_instance = raw_result.edit_DATA_conf().get_DATA_MOEA()
                     if hasattr(moea_instance, 'exec'):
                          moea_instance.exec()
                data_payload = raw_result
        except Exception as e:
            # Propagate error if needed
            raise e

        return data_payload, seed

    # Persistence
    def save(self, path: str, mode: str = 'all') -> str:
        from .save import save as legacy_save
        return legacy_save.IPL_save(self, path, mode=mode)

    def load(self, path: str, mode: str = 'all') -> None:
        from .loader import loader
        loader.IPL_loader(self, path, mode=mode)