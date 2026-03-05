# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from ..defaults import defaults
from ..core.base import Reportable

class BaseMoeaWrapper(Reportable):
    """
    Base class for MOEA wrappers that bridge between the user-facing API 
    and the internal optimization engines.
    """
    def __init__(self, engine_class, population=None, generations=None, seed=None, **kwargs):
        self._engine_class = engine_class
        
        # Use mb.defaults if not provided
        self._initial_population = population if population is not None else defaults.population
        self._initial_generations = generations if generations is not None else defaults.generations
        self._initial_seed = seed if seed is not None else defaults.seed
        self._stop = None
        self._kwargs = kwargs # Store user parameters
        self._instance = None
        self.problem = None # Set by Experiment

    def report(self, show: bool = True, **kwargs) -> str:
        """Narrative report of the algorithm configuration and parameters."""
        use_md = kwargs.get('markdown', False)
        
        # Determine algorithm name
        name = getattr(self, 'name', self.__class__.__name__)
        engine = self._engine_class.__name__ if self._engine_class else "Unknown"
        
        # Collect parameters
        params = {
            "Population": self.population,
            "Generations": self.generations,
            "Seed": self.seed
        }
        # Add extra kwargs
        for k, v in self._kwargs.items():
            params[k.capitalize()] = v
            
        if use_md:
            header = f"### Algorithm Report: {name}"
            lines = [
                header,
                f"  - **Engine**:     {engine}",
                f"  - **Status**:     {'Initialized' if self._instance else 'Configured'}",
                "",
                "#### Hyperparameters"
            ]
            for k, v in params.items():
                lines.append(f"  - **{k}**: {v}")
                
            content = "\n".join(lines)
        else:
            lines = [
                f"--- Algorithm Report: {name} ---",
                f"  Engine:     {engine}",
                f"  Status:     {'Initialized' if self._instance else 'Configured'}",
                "\n  Hyperparameters:"
            ]
            for k, v in params.items():
                lines.append(f"    {k}: {v}")
                
            content = "\n".join(lines)
            
        return self._render_report(content, show, **kwargs)

    def __call__(self, experiment, default=None, stop=None, seed=None):
        """Specializes the MOEA for a specific experiment."""
        self.problem = experiment
        
        # Priority: argument seed > instance attribute (set by setter) > initial default
        final_seed = seed if seed is not None else getattr(self, 'seed', self._initial_seed)
        
        self._instance = self._engine_class(
            experiment,
            self._initial_population,
            self._initial_generations,
            final_seed,
            stop,
            **self._kwargs # Propagate to engine
        )
        return self._instance

    def evaluation(self):
        """Standard moeabench evaluation entry point."""
        if self._instance is None:
            if self.problem is None:
                raise RuntimeError("MOEA wrapper has no experiment assigned.")
            # Use current seed property if set
            current_seed = getattr(self, 'seed', self._initial_seed)
            current_stop = getattr(self, 'stop', None)
            self.__call__(self.problem, seed=current_seed, stop=current_stop)
        else:
            # Re-initialize if seed changed? Ideally yes, but for now assuming one-shot or manual re-call
            pass
            
        return self._instance.evaluation()

    @property
    def generations(self):
        return self._initial_generations
    
    @generations.setter
    def generations(self, value):
        self._initial_generations = value
        if self._instance:
            self._instance.generations = value

    @property
    def population(self):
        return self._initial_population
    
    @population.setter
    def population(self, value):
        self._initial_population = value
        if self._instance:
            self._instance.population = value

    @property
    def seed(self):
        return self._initial_seed
    
    @seed.setter
    def seed(self, value):
        self._initial_seed = value
        if self._instance and hasattr(self._instance, 'seed'):
             self._instance.seed = value

    @property
    def stop(self):
        return self._stop
    
    @stop.setter
    def stop(self, value):
        self._stop = value
        if self._instance and hasattr(self._instance, 'stop'):
             self._instance.stop = value

    @property
    def F_gen_all(self):
        """Delegates generational objectives history from the internal engine."""
        return getattr(self._instance, 'F_gen_all', None)

    @property
    def X_gen_all(self):
        """Delegates generational decision variables history from the internal engine."""
        return getattr(self._instance, 'X_gen_all', None)
