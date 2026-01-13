# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# from .I_UserExperiment import I_UserExperiment # Legacy removed
from .run import Run, SmartArray, Population
import numpy as np
import inspect
import os
import multiprocessing
import concurrent.futures
from ..progress import get_progress_bar, set_active_pbar, ParallelProgressManager, set_worker_config
from ..system import cpus
from typing import Optional, List, Union, Any, Iterator, Dict

class JoinedPopulation:
    def __init__(self, pops: List[Population]) -> None:
        self.pops = pops
        
    @property
    def objectives(self) -> SmartArray:
        # Concatenate 
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

class experiment:
    def __init__(self, imports: Optional[Any] = None) -> None:
        self.imports = imports # Keep for compatibility if needed
        self._runs: List[Run] = []
        self._mop: Any = None
        self._moea: Any = None
        self._stop: Any = None
        self._name: str = "experiment"
        
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
        return JoinedPopulation([run.pop(gen) for run in self._runs])

    def front(self, gen: int = -1) -> SmartArray:
         # Helper for single run case or aggregate?
         # API.py: exp.front() # Front of the last run (the unique run)
         # If multiple runs, what is exp.front()? 
         # Likely the front of the LAST run (shortcut).
         res = self.last_run.front(gen)
         if hasattr(res, 'name'): res.name = self.name
         # Attach source for name inference
         if hasattr(res, 'label'): res.source = self 
         if hasattr(res, 'gen'): res.gen = res.gen # Already set by last_run.front(gen)
         return res

    def set(self, gen: int = -1) -> SmartArray:
         res = self.last_run.set(gen)
         if hasattr(res, 'name'): res.name = self.name
         if hasattr(res, 'label'): res.source = self
         if hasattr(res, 'gen'): res.gen = res.gen # Already set
         return res

    def all_fronts(self, gen: int = -1) -> List[SmartArray]:
        """Returns a list of Pareto fronts from all runs."""
        return [run.front(gen) for run in self._runs]

    def all_sets(self, gen: int = -1) -> List[SmartArray]:
        """Returns a list of decision sets from all runs."""
        return [run.set(gen) for run in self._runs]

    def superfront(self, gen: int = -1) -> SmartArray:
        """Returns the non-dominated front considering all runs combined."""
        p = self.pop(gen)
        # Create a combined population to apply global filtering
        combined = Population(p.objectives, p.variables, source=self, label="Superfront")
        return combined.non_dominated().objectives

    def superset(self, gen: int = -1) -> SmartArray:
        """Returns the non-dominated decision set considering all runs combined."""
        p = self.pop(gen)
        combined = Population(p.objectives, p.variables, source=self, label="Superset")
        return combined.non_dominated().variables

    def non_front(self, gen: int = -1) -> SmartArray:
         res = self.last_run.non_front(gen)
         if hasattr(res, 'name'): res.name = self.name
         if hasattr(res, 'label'): res.source = self
         return res

    def non_set(self, gen: int = -1) -> SmartArray:
         res = self.last_run.non_set(gen)
         if hasattr(res, 'name'): res.name = self.name
         if hasattr(res, 'label'): res.source = self
         return res

    def dominated(self, gen: int = -1) -> Population:
         """Returns the dominated Population at gen."""
         pop = self.last_run.dominated(gen)
         pop.label = "Dominated"
         
         # Inject metadata for automatic naming
         for arr in [pop.objectives, pop.variables]:
             if hasattr(arr, 'name'): arr.name = self.name
             if hasattr(arr, 'source'): arr.source = self
             arr.label = "Dominated"
             if hasattr(arr, 'gen'): arr.gen = pop.gen # Propagate
             
         return pop

    def non_dominated(self, gen: int = -1) -> Population:
         """Returns the non-dominated Population at gen."""
         pop = self.last_run.non_dominated(gen)
         pop.label = "Non-dominated"

         # Inject metadata for automatic naming
         for arr in [pop.objectives, pop.variables]:
             if hasattr(arr, 'name'): arr.name = self.name
             if hasattr(arr, 'source'): arr.source = self
             arr.label = "Non-dominated"
             if hasattr(arr, 'gen'): arr.gen = pop.gen # Propagate
             
         return pop

    def optimal(self, n_points: int = 500) -> Population:
        """Returns a sampling of the true Pareto optimal set and front."""
        if not hasattr(self.mop, 'pf'):
             raise AttributeError(f"MOP {self.mop.__class__.__name__} does not implement pf() sampling.")
        
        objs = self.mop.pf(n_points)
        vars = self.mop.ps(n_points)
        
        # Create Population with "Optimal" label
        # We pass self as source so metadata (like experiment name) is available,
        # and label="Optimal" to indicate this is the theoretical PF/PS.
        pop = Population(objs, vars, source=self, label="Optimal")
        
        # For analytical optimal fronts, we want the name to be "Optimal" 
        # instead of the experiment name (NSGA3, etc.)
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
         
    # Execution
    def run(self, repeat: int = 1, workers: Optional[int] = None) -> None:
        """
        Executes the optimization experiment for one or more runs.

        Args:
            repeat (int): Number of independent runs to perform.
            workers (int): Number of parallel workers to use. 
                          If None, execution is serial. 
                          If -1, uses all available CPU cores.
        """
        if repeat < 1: repeat = 1
        
        # Determine base seed
        base_seed = getattr(self.moea, 'seed', None)
        if base_seed is None:
            base_seed = np.random.randint(0, 1000000)
            if hasattr(self.moea, 'seed'):
                 self.moea.seed = base_seed

        # Setup Parallelism
        if workers == -1:
            workers = cpus(safe=True)
        elif workers == 0:
            workers = max(1, cpus() // 2)
        
        if workers and workers > 1:
            self._run_parallel(repeat, workers, base_seed)
        else:
            self._run_serial(repeat, base_seed)

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
                run_data, _, _ = _moea_run_worker(self.moea, self.mop, seed, i+1)
                new_run = Run(run_data, seed, experiment=self, index=i+1)
                self._runs.append(new_run)
            finally:
                inner_pbar.close()
                set_active_pbar(None)
                if outer_pbar:
                    outer_pbar.update_to(i + 1)

        if outer_pbar:
            outer_pbar.close()

    def _run_parallel(self, repeat: int, workers: int, base_seed: int) -> None:
        # Create manager and queue for progress reporting
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        progress_mgr = ParallelProgressManager(repeat=repeat, desc=f"Experiment: {self.name} (Parallel)", 
                                                 workers=workers, queue=progress_queue)
        progress_mgr.start()

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                # Schedule all runs
                futures = []
                for i in range(repeat):
                    seed = base_seed + i
                    futures.append(executor.submit(_moea_run_worker, self.moea, self.mop, seed, i+1, progress_queue))
                
                # Collect results as they finish
                results = [None] * repeat
                for future in concurrent.futures.as_completed(futures):
                    data, seed, index = future.result()
                    results[index-1] = (data, seed)
                
                # Build Run objects in correct order
                for i in range(repeat):
                    data, seed = results[i]
                    new_run = Run(data, seed, experiment=self, index=i+1)
                    self._runs.append(new_run)
        finally:
            progress_mgr.stop()

    # Persistence
    def save(self, path: str) -> None:
        # Reuse legacy save if possible, or implement simple pickle
        from .save import save as legacy_save
        legacy_save.IPL_save(self, path)

    def load(self, path: str) -> None:
        from .loader import loader
        loader.IPL_loader(self, path)

def _moea_run_worker(moea, mop, seed, index, queue=None):
    """Standalone worker function for ProcessPoolExecutor."""
    if queue:
        set_worker_config(queue, index)
    
    # Inject context
    moea.problem = mop 
    if hasattr(moea, 'seed'):
        moea.seed = seed
    
    # Progress bar for this run
    total_gens = getattr(moea, 'generations', None)
    # Each worker gets its own progress bar that redirects to queue
    pbar = get_progress_bar(total=total_gens, desc=f"  Run {index}", leave=False)
    set_active_pbar(pbar)

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
    finally:
        pbar.close()
        set_active_pbar(None)

    return data_payload, seed, index