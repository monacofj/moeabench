# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from typing import Optional, List, Union, Tuple, Any

class SmartArray(np.ndarray):
    """
    Numpy array wrapper that carries metadata about the data it holds (label, axis_label, name).
    """
    label: Optional[str]
    axis_label: Optional[str]
    name: Optional[str]
    source: Any
    gen: Optional[int]

    def __new__(cls, input_array: Any, label: Optional[str] = None, 
                axis_label: Optional[str] = None, name: Optional[str] = None, 
                source: Any = None, gen: Optional[int] = None) -> 'SmartArray':
        obj = np.asarray(input_array).view(cls)
        obj.label = label
        obj.axis_label = axis_label
        obj.gen = gen
        
        # Priority: explicit name > source.name > None
        if name is None and source is not None:
            name = getattr(source, 'name', None)
            
        obj.name = name 
        obj.source = source
        return obj

    def __array_finalize__(self, obj: Optional[Any]) -> None:
        if obj is None: return
        self.label = getattr(obj, 'label', None)
        self.axis_label = getattr(obj, 'axis_label', None)
        self.name = getattr(obj, 'name', None)
        self.source = getattr(obj, 'source', None)
        self.gen = getattr(obj, 'gen', None)

class Run:
    """
    Represents a single trajectory (one run) of an evolutionary algorithm.
    
    Stores the generational history of populations, including objectives, 
    decision variables, and non-dominated subsets.
    """
    def __init__(self, data_payload: Optional[Tuple] = None, seed: Optional[int] = None, 
                 experiment: Any = None, index: Optional[int] = None) -> None:
        """
        Initializes a Run object.

        Args:
            data_payload (tuple): A tuple containing (F_gens, X_gens, F_final, 
                                 hist_F_nd, hist_X_nd, hist_F_dom, hist_X_dom).
            seed (int): The random seed used for this run.
            experiment: The parent Experiment object.
            index (int): 1-based index of this run within the experiment.
        """
        self.seed = seed
        self.source = experiment
        self.index = index
        
        # New direct storage
        if data_payload:
            self._F_history: List[np.ndarray] = data_payload[0]
            self._X_history: List[np.ndarray] = data_payload[1]
            self._F_final: Optional[np.ndarray] = data_payload[2]
            self._F_nd_history: List[np.ndarray] = data_payload[3]
            self._X_nd_history: List[np.ndarray] = data_payload[4]
            self._F_dom_history: List[np.ndarray] = data_payload[5]
            self._X_dom_history: List[np.ndarray] = data_payload[6]
        else:
            self._F_history = []
            self._X_history = []
            self._F_final = None
            self._F_nd_history = []
            self._X_nd_history = []
            self._F_dom_history = []
            self._X_dom_history = []

    @property
    def name(self) -> Optional[str]:
        """str: The identifying name of this run for plot legends."""
        base_name = None
        if self.source:
            base_name = getattr(self.source, 'name', None)
        
        if base_name and self.index is not None and self.source and len(self.source) > 1:
            return f"{base_name} (run {self.index})"
        return base_name

    def __repr__(self) -> str:
        return f"<Run generations={len(self)}>"

    def __len__(self) -> int:
        """int: Number of generations in this run."""
        return len(self._F_history)

    def history(self, type: str = 'nd') -> List[np.ndarray]:
        """
        Returns a trajectory (list of arrays) of population data over time.

        Args:
            type (str): 'nd' (non-dominated objectives), 'f' (all objectives), 
                       'x' (all variables), 'nd_x' (non-dominated variables),
                       'dom' (dominated objectives), 'dom_x' (dominated variables).
        """
        if type == 'nd': return self._F_nd_history
        if type == 'f': return self._F_history
        if type == 'x': return self._X_history
        if type == 'nd_x': return self._X_nd_history
        if type == 'dom': return self._F_dom_history
        if type == 'dom_x': return self._X_dom_history
        raise ValueError(f"Unknown history type: {type}")

    def pop(self, gen: int = -1) -> 'Population':
        """
        Returns the population at a specific generation.

        Args:
            gen (int): Generation index (0-based). Default is -1.

        Returns:
            Population: The population object for the given generation.
        """
        if gen < 0:
            gen = len(self) + gen
            
        if not (0 <= gen < len(self)):
            raise IndexError(f"Generation index {gen} out of range (total {len(self)})")
            
        objs = self._F_history[gen]
        vars = self._X_history[gen]
        
        return Population(objs, vars, source=self, gen=gen)

    def front(self, gen: int = -1) -> Union[SmartArray, Any]:
        """
        Returns the non-dominated objectives (Pareto Front) at a generation.

        Args:
            gen (int): Generation index. Default is -1.

        Returns:
            SmartArray: The Pareto front as an objective matrix.
        """
        if gen < 0:
            gen = len(self._F_nd_history) + gen
            
        if 0 <= gen < len(self._F_nd_history):
            return SmartArray(self._F_nd_history[gen], 
                              label="Pareto Front", 
                              axis_label="Objective", 
                              source=self, 
                              gen=gen)
        
        # Fallback to dynamic calculation if history missing
        return self.non_dominated(gen).objectives

    def set(self, gen: int = -1) -> Union[SmartArray, Any]:
        """Returns the non-dominated decision variables at a generation."""
        if gen < 0:
            gen = len(self._X_nd_history) + gen
            
        if 0 <= gen < len(self._X_nd_history):
            return SmartArray(self._X_nd_history[gen], 
                              label="Pareto Set", 
                              axis_label="Variable", 
                              source=self, 
                              gen=gen)
        
        return self.non_dominated(gen).variables

    def non_front(self, gen: int = -1) -> Union[SmartArray, Any]:
        """Returns the dominated objectives at a generation."""
        if gen < 0:
            gen = len(self._F_dom_history) + gen
            
        if 0 <= gen < len(self._F_dom_history):
            return SmartArray(self._F_dom_history[gen], 
                              label="Dominated Front", 
                              axis_label="Objective", 
                              source=self, 
                              gen=gen)
        
        return self.dominated(gen).objectives

    def non_set(self, gen: int = -1) -> Union[SmartArray, Any]:
        """Returns the dominated decision variables at a generation."""
        if gen < 0:
            gen = len(self._X_dom_history) + gen
            
        if 0 <= gen < len(self._X_dom_history):
            return SmartArray(self._X_dom_history[gen], 
                              label="Dominated Set", 
                              axis_label="Variable", 
                              source=self, 
                              gen=gen)
        
        return self.dominated(gen).variables
        
    def non_dominated(self, gen: int = -1) -> 'Population':
        """Returns the non-dominated Population at a generation."""
        return self.pop(gen).non_dominated()
        
    def dominated(self, gen: int = -1) -> 'Population':
        """Returns the dominated Population at a generation."""
        return self.pop(gen).dominated()

    @property
    def last_pop(self) -> 'Population':
        """Population: Shortcut for the final generation's population."""
        return self.pop()


class Population:
    """
    Container for individuals at a specific point in the search.
    
    Provides methods for non-dominance filtering and data access.
    """
    def __init__(self, objectives: np.ndarray, variables: np.ndarray, 
                 source: Any = None, label: str = "Population", gen: Optional[int] = None) -> None:
        """
        Initializes a Population.

        Args:
            objectives (np.ndarray): Objective matrix (N x M).
            variables (np.ndarray): Decision variable matrix (N x D).
            source: The parent Run or Experiment.
            label (str): Label for plotting.
            gen (int): Generation index.
        """
        self.objectives = SmartArray(objectives, label=label, axis_label="Objective", source=source, gen=gen)
        self.variables = SmartArray(variables, label=label, axis_label="Variable", source=source, gen=gen)
        self.source = source
        self.label = label
        self.gen = gen
        
    @property
    def objs(self) -> SmartArray: return self.objectives
    
    @property
    def vars(self) -> SmartArray: return self.variables

    def __len__(self) -> int:
        return self.objectives.shape[0]

    def non_dominated(self) -> 'Population':
        """Returns a new Population containing only non-dominated solutions."""
        is_dominated = self._calc_domination()
        return Population(self.objectives[~is_dominated], self.variables[~is_dominated], 
                          source=self.source, label=self.label, gen=self.gen)
    
    def dominated(self) -> 'Population':
        """Returns a new Population containing only dominated solutions."""
        is_dominated = self._calc_domination()
        return Population(self.objectives[is_dominated], self.variables[is_dominated], 
                          source=self.source, label=self.label, gen=self.gen)

    def _calc_domination(self) -> np.ndarray:
        """
        Computes dominance status for all individuals (minimization assumed).
        Uses a chunked broadcasting approach to balance speed and memory usage.
        
        Returns:
            np.ndarray: Boolean mask where True means 'dominated'.
        """
        objs = self.objectives
        N = objs.shape[0]
        
        if N == 0:
            return np.zeros(0, dtype=bool)
            
        # Determine chunk size to avoid memory explosion (O(N^2))
        # A chunk size of 500-1000 is usually a good balance.
        # N * chunk_size * 8 bytes (float64) should fit comfortably in RAM.
        chunk_size = 500
        is_dominated = np.zeros(N, dtype=bool)
        
        # We compare everyone against everyone, but in chunks.
        # A (potential dominators): (1, N, M)
        # B (potential victims):    (chunk, 1, M)
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            B = objs[start:end, np.newaxis, :] # (chunk, 1, M)
            A = objs[np.newaxis, :, :]         # (1, N, M)
            
            # j (from A) dominates i (from B) if (j <= i for all M) AND (j < i for at least one M)
            dominates_all = np.all(A <= B, axis=2) 
            dominates_any = np.any(A < B, axis=2)  
            
            # Matrix of who dominates who in this chunk
            # entry [i, j] is True if 'j' dominates 'i'
            dominance_matrix = dominates_all & dominates_any
            
            # Solution 'i' in the chunk is dominated if ANY 'j' in the whole population dominates it
            is_dominated[start:end] = np.any(dominance_matrix, axis=1)
            
        return is_dominated

    def stratify(self) -> np.ndarray:
        """
        Performs full Non-Dominated Sorting (NDS) by iteratively peeling 
        non-dominated layers.
        
        This implementation leverages the optimized chunked broadcasting 
        logic from _calc_domination but applies it recursively to discover 
        all ranks.
        
        Returns:
            np.ndarray: Integer array where each entry is the rank (1-indexed) 
                       of the corresponding individual.
        """
        N = len(self)
        if N == 0:
            return np.zeros(0, dtype=int)
            
        ranks = np.zeros(N, dtype=int)
        remaining_mask = np.ones(N, dtype=bool)
        current_rank = 1
        
        full_objs = self.objectives
        
        while np.any(remaining_mask):
            # Indices of individuals we are still ranking
            indices = np.where(remaining_mask)[0]
            sub_objs = full_objs[indices]
            
            # Find which individuals in the remaining set are non-dominated 
            # WITH RESPECT TO the remaining set.
            # We reuse the logic: i is dominated if some j in ONLY THE REMAINING SET dominates it.
            sub_N = len(indices)
            sub_is_dominated = np.zeros(sub_N, dtype=bool)
            
            # Chunked check within the sub-set
            chunk_size = 500
            for start in range(0, sub_N, chunk_size):
                end = min(start + chunk_size, sub_N)
                B = sub_objs[start:end, np.newaxis, :] 
                A = sub_objs[np.newaxis, :, :]         
                
                dominance_matrix = np.all(A <= B, axis=2) & np.any(A < B, axis=2)
                sub_is_dominated[start:end] = np.any(dominance_matrix, axis=1)
            
            # Those NOT dominated in the sub-set are the current rank
            current_rank_sub_indices = np.where(~sub_is_dominated)[0]
            actual_indices = indices[current_rank_sub_indices]
            
            ranks[actual_indices] = current_rank
            remaining_mask[actual_indices] = False
            current_rank += 1
            
        return ranks
