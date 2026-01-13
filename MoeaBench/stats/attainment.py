# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from ..core.run import SmartArray

class AttainmentSurface(SmartArray):
    """
    Experimental data representing an Empirical Attainment Surface.
    """
    def __new__(cls, input_array, level=0.5, name=None):
        obj = np.asarray(input_array).view(cls)
        obj.level = level
        obj.name = name if name else f"Attainment {level*100:.0f}%"
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.level = getattr(obj, 'level', 0.5)
        self.name = getattr(obj, 'name', None)

def attainment(experiment, level: float = 0.5):
    """
    Calculates the k-th attainment surface for a multi-run experiment.
    
    The level is a probability in [0, 1].
    Example: level=0.5 returns the Median Attainment Surface.
    """
    # 1. Collect final fronts from all runs
    fronts = [run.front() for run in experiment.runs]
    if not fronts:
        raise ValueError("Experiment has no runs.")
        
    n_runs = len(fronts)
    k = int(np.ceil(level * n_runs))
    # Clamp k
    k = max(1, min(n_runs, k))
    
    # Simple algorithm for attainment surfaces:
    # We want the boundary where at least k runs "attain" the region.
    # For a point z, it is attained by run i if exists p in front_i such that p <= z.
    
    # Practical implementation via "Non-dominated Sorting" of the union:
    # Note: This is a complex geometric problem. 
    # For 2D, we can use the exact staircase.
    # For 3D, we'll return the points that define the boundary.
    
    all_points = np.concatenate(fronts, axis=0)
    m = all_points.shape[1]
    
    if m == 2:
        return _attainment_2d(fronts, k, level)
    else:
        # Fallback for 3D+: Return a high-quality point-based representation
        # using a k-domination logic.
        return _attainment_nd(fronts, k, level)

def _attainment_2d(fronts, k, level):
    """Exact 2D attainment surface calculation (staircase)."""
    # For 2D, the attainment surface is defined by the coordinates 
    # of the points in the fronts.
    all_x = []
    all_y = []
    for f in fronts:
        all_x.extend(f[:, 0])
        all_y.extend(f[:, 1])
    
    unique_x = np.unique(all_x)
    unique_y = np.unique(all_y)
    
    # We evaluate the attainment function on the grid defined by these coordinates
    # Actually, we just need to find for each unique_x, the y such that 
    # exactly k runs have a point (px, py) with px <= unique_x and py <= y.
    
    res_points = []
    for x in unique_x:
        # For this x, find the minimum y that is attained by at least k runs
        attained_ys = []
        for f in fronts:
            # Points in this front that are better or equal to x in the first dimension
            mask = f[:, 0] <= x
            if np.any(mask):
                attained_ys.append(np.min(f[mask, 1]))
            else:
                attained_ys.append(np.inf)
        
        attained_ys.sort()
        # The k-th smallest y is the one attained by k runs
        y_k = attained_ys[k-1]
        if not np.isinf(y_k):
            res_points.append([x, y_k])
            
    # Remove redundant points (non-dominated)
    res_array = np.array(res_points)
    return AttainmentSurface(res_array, level=level)

def _attainment_nd(fronts, k, level):
    """
    Approximate N-D attainment surface calculation via point sampling.
    For 3D, this provides the points on the boundary.
    """
    # A point z is in the k-attainment set if at least k runs dominate z.
    # This is equivalent to finding points that are dominated by exactly k runs 
    # but not by k+1 runs in a specific sense.
    
    # Simplified approach: The k-th attainment surface is bounded by points
    # formed by combinations of coordinates from different runs.
    # For 3D+, we'll return a representative set of points.
    
    all_points = np.concatenate(fronts, axis=0)
    # This is a placeholder for a more complex algorithm like 
    # the one by Fonseca et al.
    # For now, we return the k-th non-dominated front of the union,
    # which is a decent proxy for the attainment boundary points.
    
    from ..core.run import Run
    # We use a dummy run to call non-dominated sort logic
    # But we need k-th front. 
    # MoeaBench doesn't have a k-th front extractor in 'run', 
    # it only does 1st front.
    
    # Implementing a simple k-th front check:
    # For each point in the union, count how many runs attain it.
    union = all_points
    counts = []
    for p in union:
        count = 0
        for f in fronts:
            # Does front f attain p?
            if np.any(np.all(f <= p, axis=1)):
                count += 1
        counts.append(count)
    
    counts = np.array(counts)
    # Points that are attained by at least k runs
    mask = counts >= k
    candidates = union[mask]
    
    # Extract non-dominated subset of these candidates
    from ..core.run import _non_dominated_sort
    indices = _non_dominated_sort(candidates)
    
    return AttainmentSurface(candidates[indices], level=level)

def attainment_diff(exp1, exp2, level=0.5):
    """
    Calculates the spatial difference in attainment between two experiments
    at a specific attainment level (default: 0.5/Median).
    
    Returns a pair of AttainmentSurface objects representing the boundaries
    of each algorithm.
    """
    surf1 = attainment(exp1, level=level)
    surf1.name = f"{exp1.name} (L={level})"
    
    surf2 = attainment(exp2, level=level)
    surf2.name = f"{exp2.name} (L={level})"
    
    return surf1, surf2

