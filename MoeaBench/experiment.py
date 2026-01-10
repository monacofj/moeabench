# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .I_UserExperiment import I_UserExperiment
from .Run import Run, SmartArray
from .RUN import RUN
from .RUN_user import RUN_user
import numpy as np
import inspect

class experiment(I_UserExperiment):
    def __init__(self, imports=None):
        self.imports = imports # Keep for compatibility if needed
        self._runs = []
        self._benchmark = None
        self._moea = None
        self._stop = None
        self._name = "experiment"
        
        # Internal state for execution
        self.result = None 

    def __iter__(self):
        return iter(self._runs)

    def __getitem__(self, index):
        return self._runs[index]

    def __len__(self):
        return len(self._runs)

    @property
    def name(self): return self._name
    @name.setter
    def name(self, value): self._name = value
    
    @property
    def benchmark(self): return self._benchmark
    @benchmark.setter
    def benchmark(self, value):
        # Auto-instantiate if it's a factory (callable but not the final instance with get_CACHE)
        # We assume if it's callable and missing core traits, it's a factory.
        if callable(value) and not hasattr(value, 'get_CACHE'):
             try:
                 self._benchmark = value()
             except Exception as e:
                 # Fallback or error logging
                 print(f"Warning: Could not instantiate benchmark factory: {e}")
                 self._benchmark = value
        else:
            self._benchmark = value

    @property
    def moea(self): return self._moea
    @moea.setter
    def moea(self, value):
        self._moea = value
        # Original code did magic here to instantiate result.
        # We will handle instantiation in run()

    @property
    def stop(self): return self._stop
    @stop.setter
    def stop(self, value): self._stop = value
    
    @property
    def pof(self):
        """Legacy compatibility for save mechanism."""
        return self._benchmark

    # Shortcuts
    @property
    def last_run(self):
        if not self._runs: raise IndexError("No runs executed yet")
        return self._runs[-1]

    @property
    def last_pop(self):
        return self.last_run.last_pop

    def pop(self, gen=-1):
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

    def front(self, gen=-1):
         # Helper for single run case or aggregate?
         # API.py: exp.front() # Front of the last run (the unique run)
         # If multiple runs, what is exp.front()? 
         # Likely the front of the LAST run (shortcut).
         res = self.last_run.front(gen)
         if hasattr(res, 'name'): res.name = self.name
         # Attach source for name inference
         if hasattr(res, 'label'): res.source = self 
         return res

    def set(self, gen=-1):
         res = self.last_run.set(gen)
         if hasattr(res, 'name'): res.name = self.name
         if hasattr(res, 'label'): res.source = self
         return res

    def dominated(self, gen=-1):
         """Returns the dominated objectives at gen."""
         # Similar to front(), return array of dominated objectives
         pop = self.last_run.dominated(gen)
         res = pop.objectives # Already a SmartArray from Population
         
         # Overwrite label to be specific
         res.label = "Dominated Solutions"
         
         if hasattr(res, 'name'): res.name = self.name
         if hasattr(res, 'label'): res.source = self
         return res

    def nondominated(self, gen=-1):
         """Returns the non-dominated Population at gen."""
         pop = self.last_run.nondominated(gen)
         
         # Inject metadata into population's arrays for convenience
         if hasattr(pop.objectives, 'name'): pop.objectives.name = self.name
         if hasattr(pop.objectives, 'label'): pop.objectives.source = self
         if hasattr(pop.variables, 'name'): pop.variables.name = self.name
         if hasattr(pop.variables, 'label'): pop.variables.source = self
         
         return pop

    # Shortcuts from API.py
    @property
    def objectives(self):
        return self.pop().objectives

    @property
    def variables(self):
        return self.pop().variables
         
    # Execution
    def run(self, repeat=1):
        if repeat < 1: repeat = 1
        
        # Instantiate runner mechanism
        # This part is preserving the legacy "RUN" logic which is complex
        # We need to adapt it to produce a Run object.
        
        for _ in range(repeat):
            seed = np.random.randint(0, 1000000) # Simple seeder
            
            # Legacy logic adaptation:
            # We need to "setup" the MoEA. 
            # Original: self.result = value(self, self.imports.moeas, stop)
            
            # Let's try to mimic the essential parts of run_moea
            # We assume self.moea is the algorithm OBJECT (e.g. initialised NSGA3)
            # We need to run it on self.benchmark
            
            # If the architecture requires using the RUN class:
            # execute = RUN()
            # self.result = self.moea(self, None, self.stop, seed)
            
            # The moea object in moeabench seems to be callable?
            # exp.moea = mb.moeas.NSGA3()
            # NSGA3 is an object. Is it callable?  
            # Base logic seems to be: result = moea_instance(experiment_ctx, ...)
            
            # Let's trust the 'moea' object is capable of running
            # In original `run_moea`:
            # self.result = self.moea(self, None, stop, seed) 
            
            # We pass 'self' (experiment). The MOEA might update self.result?
            # Or return it.
            
            # Wrapper returns setup CACHE
            current_result = self.moea(self, None, self.stop, seed)
            
            # Trigger execution!
            # The result from 'self.moea' is a CACHE object (or tuple wrapping it)
            raw_result = current_result[0] if isinstance(current_result, tuple) else current_result
            
            # Legacy support: Execution callback expects exp.result to be the current cache
            self.result = raw_result
            
            if hasattr(raw_result, 'edit_DATA_conf'):
                 moea_instance = raw_result.edit_DATA_conf().get_DATA_MOEA()
                 if hasattr(moea_instance, 'exec'):
                      moea_instance.exec()
                 else:
                      print("Warning: MOEA instance does not have exec() method.")
            
            # Create Run wrapper
            new_run = Run(raw_result, seed)
            self._runs.append(new_run)

    # Persistence
    def save(self, path):
        # Reuse legacy save if possible, or implement simple pickle
        from .save import save as legacy_save
        legacy_save.IPL_save(self, path)

    def load(self, path):
        from .loader import loader
        loader.IPL_loader(self, path)
        # Note: loader populates self.result usually. 
        # We need to extract runs from it? 
        # Legacy loader likely restores the old state structure. 
        # This is a risk point. For now, assume loader works enough to pop result.

class JoinedPopulation:
    def __init__(self, pops):
        self.pops = pops
        
    @property
    def objectives(self):
        # Concatenate 
        data = np.vstack([p.objectives for p in self.pops])
        return SmartArray(data, label="Population (Objectives)", axis_label="Objective")
        
    @property
    def variables(self):
        data = np.vstack([p.variables for p in self.pops])
        return SmartArray(data, label="Population (Variables)", axis_label="Variable")