class BaseMoeaWrapper:
    """
    Base class for MOEA wrappers that bridge between the user-facing API 
    and the internal optimization engines.
    """
    def __init__(self, engine_class, population=150, generations=300, seed=1):
        self._engine_class = engine_class
        self._initial_population = population
        self._initial_generations = generations
        self._initial_seed = seed
        self._instance = None
        self.problem = None # Set by Experiment

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
            stop
        )
        return self._instance

    def evaluation(self):
        """Standard MoeaBench evaluation entry point."""
        if self._instance is None:
            if self.problem is None:
                raise RuntimeError("MOEA wrapper has no experiment assigned.")
            # Use current seed property if set
            current_seed = getattr(self, 'seed', self._initial_seed)
            self.__call__(self.problem, seed=current_seed)
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
