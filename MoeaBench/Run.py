from .result_population import result_population
import numpy as np

class SmartArray(np.ndarray):
    """
    Numpy array wrapper that carries metadata about the data it holds (label, axis_label, name).
    """
    def __new__(cls, input_array, label=None, axis_label=None, name=None):
        obj = np.asarray(input_array).view(cls)
        obj.label = label
        obj.axis_label = axis_label
        obj.name = name # Name of the dataset (e.g. Experiment name)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.label = getattr(obj, 'label', None)
        self.axis_label = getattr(obj, 'axis_label', None)
        self.name = getattr(obj, 'name', None)

class Run:
    """
    Represents a single execution (trajectory) of an optimization algorithm.
    Provides access to populations across generations.
    """
    def __init__(self, engine_result, seed=None):
        """
        Args:
            engine_result: The internal result object from the MOEA engine 
                           (must support get_F_GEN, get_X_GEN, etc.)
            seed: The random seed used for this run.
        """
        self.seed = seed
        
        # Unwrap CACHE object to find the actual data container (DATA_conf)
        self._engine_result = self._find_data_conf(engine_result)

    def _find_data_conf(self, res):
        # res is likely a CACHE object
        if hasattr(res, 'get_elements'):
             elements = res.get_elements()
             # elements is list of lists
             for group in elements:
                 for group_item in group:
                     # Check if it's the item or inside the item?
                     # Legacy code iterated groups then items.
                     # Let's assume group_item IS the item
                     if hasattr(group_item, 'get_F_GEN'):
                         return group_item
        # Fallback: maybe it IS the data conf?
        if hasattr(res, 'get_F_GEN'):
            return res
            
        raise ValueError(f"Could not find DATA_conf with 'get_F_GEN' in result: {res}")
            
    @property
    def result(self):
        return self._engine_result

    def __repr__(self):
        return f"<Run generations={len(self)}>"

    def __len__(self):
        # Assuming get_F_GEN returns a list of generations
        return len(self._engine_result.get_F_GEN())

    def pop(self, gen=-1):
        """
        Returns the population at the specified generation.
        Args:
            gen: Generation index (0-based). Default is -1 (last).
        """
        # Handle negative indices
        if gen < 0:
            gen = len(self) + gen
            
        objs = self._engine_result.get_F_GEN()[gen]
        vars = self._engine_result.get_X_GEN()[gen]
        
        return Population(objs, vars)

    def front(self, gen=-1):
        """Returns the non-dominated objectives (Pareto front) at gen."""
        # For efficiency, if the engine already calculates this, use it.
        if hasattr(self._engine_result, 'get_F_gen_non_dominate'):
             fronts = self._engine_result.get_F_gen_non_dominate()
             if gen < 0: gen = len(fronts) + gen
             if 0 <= gen < len(fronts):
                 return SmartArray(fronts[gen], label="Pareto Front", axis_label="Objective")
        
        # Fallback
        return self.pop(gen).nondominated().objectives

    def set(self, gen=-1):
        """Returns the non-dominated variables (Pareto set) at gen."""
        if hasattr(self._engine_result, 'get_X_gen_non_dominate'):
             sets = self._engine_result.get_X_gen_non_dominate()
             if gen < 0: gen = len(sets) + gen
             if 0 <= gen < len(sets):
                 return SmartArray(sets[gen], label="Pareto Set", axis_label="Variable")
        
        return self.pop(gen).nondominated().variables

    def nondominated(self, gen=-1):
        """Returns the non-dominated Population at gen."""
        return self.pop(gen).nondominated()
        
    def dominated(self, gen=-1):
        """Returns the dominated Population at gen."""
        return self.pop(gen).dominated()

    @property
    def last_pop(self):
        """Shortcut for the last population."""
        return self.pop()


class Population:
    """
    Represents a population of solutions (vectors in Objective and Decision space).
    """
    def __init__(self, objectives, variables):
        self.objectives = SmartArray(objectives, label="Population (Objectives)", axis_label="Objective")
        self.variables = SmartArray(variables, label="Population (Variables)", axis_label="Variable")
        
    @property
    def objs(self): return self.objectives
    
    @property
    def vars(self): return self.variables

    def __len__(self):
        return self.objectives.shape[0]

    def nondominated(self):
        """Returns a new Population containing only non-dominated solutions."""
        is_dominated = self._calc_domination()
        # Filter (keep those NOT dominated)
        # Note: Indexing SmartArray returns SmartArray (via __array_finalize__)
        # But we create a new Population, which re-wraps them.
        return Population(self.objectives[~is_dominated], self.variables[~is_dominated])
    
    def dominated(self):
         """Returns a new Population containing only dominated solutions."""
         is_dominated = self._calc_domination()
         return Population(self.objectives[is_dominated], self.variables[is_dominated])

    def _calc_domination(self):
        """
        Calculates dominance status for each individual.
        Returns Boolean array where True means the individual IS DOMINATED by someone else.
        Assumes minimization.
        """
        N = self.objectives.shape[0]
        is_dominated = np.zeros(N, dtype=bool)
        
        # Simple O(N^2) comparison
        # Optimization: use broadcasting if N is small, or loop if large to save memory.
        # Given typically small populations (N ~100-1000) in MOEA, broadcasting is likely fine 
        # but N^2 boolean matrix might be heavy if N=10k.
        # Let's use loop for safety or broadcasting if reasonable?
        # N=100 -> 10k bools (safe).
        
        # Broadcasting approach:
        # P1: (N, 1, M)
        # P2: (1, N, M)
        # P2 dominates P1?
        # all(P2 <= P1) AND any(P2 < P1)
        
        objs = self.objectives
        
        # Expand dims
        # A dominates B if A <= B for all obj, and A < B for at least one
        # We want to find for each i (target), is there any j (source) that dominates i?
        
        # Rows = source (j), Cols = target (i)
        # We check column-wise if any row dominates it.
        
        # Broadcast: (N, 1, M) vs (1, N, M) => (N, N, M) Ouch memory for large N.
        # Let's stick to simple efficient loops or slightly better array ops.
        
        # Try a compromise: Loop over individuals to check if they are dominated
        for i in range(N):
            current = objs[i]
            # Compare against ALL others
            # strictly_worse: current >= other (in all) & current > other (in at least one)
            # Wait, we want to know if 'current' IS DOMINATED by 'other'.
            # 'other' dominates 'current' if:
            # other <= current (all) AND other < current (any)
            
            # Using broadcasting for one-vs-all:
            # others <= current: (N, M) <= (M,) -> (N, M) bool -> (N,) all()
            dominates_all = np.all(objs <= current, axis=1)
            dominates_any = np.any(objs < current, axis=1)
            
            dominators = dominates_all & dominates_any
            
            # Exclude self?
            # if j == i: other <= current is True, other < current is False.
            # So dominates_any handles self-check (False).
            
            if np.any(dominators):
                is_dominated[i] = True
                
        return is_dominated

