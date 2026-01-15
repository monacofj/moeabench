# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from ..core.run import SmartArray
from .base import StatsResult


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

    def volume(self, ref_point=None):
        """
        Calculates the area (2D) or hypervolume (3D+) attained by this surface.
        """
        from ..metrics.evaluator import hypervolume
        from ..core.run import Run
        
        # Pass self directly; hypervolume() handles np.ndarray
        hv_matrix = hypervolume(self, ref=ref_point)
        return float(hv_matrix)

def attainment(source, level: float = 0.5):
    """
    Calculates the k-th attainment surface for a multi-run experiment 
    or a list of pre-calculated Pareto fronts.
    
    The level is a probability in [0, 1].
    Example: level=0.5 returns the Median Attainment Surface.
    """
    # 1. Collect final fronts
    if isinstance(source, list):
        fronts = source
    else:
        # Assume it's an Experiment object
        fronts = [run.front() for run in source.runs]

    if not fronts:
        raise ValueError("No fronts available for attainment calculation.")
        
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
    N-D attainment surface calculation.
    Uses chunked vectorized counting to find points attained by at least k runs.
    """
    # A point z is in the k-attainment set if at least k runs dominate z.
    # We use the union of all points as candidates for the boundary.
    union = np.concatenate(fronts, axis=0)
    
    attainment_counts = np.zeros(len(union), dtype=int)
    chunk_size = 500
    
    for f in fronts:
        # Front f attains point p if any q in f is <= p (all dimensions).
        # We loop over union in chunks to keep memory usage low.        
        for start in range(0, len(union), chunk_size):
            end = min(start + chunk_size, len(union))
            chunk = union[start:end]
            
            # Logic: exists q in f such that q <= p
            # (chunk, 1, M) compared with (1, N_f, M) -> (chunk, N_f, M)
            is_attained = np.any(np.all(f[np.newaxis, :, :] <= chunk[:, np.newaxis, :], axis=2), axis=1)
            attainment_counts[start:end] += is_attained.astype(int)
    
    # Points that are attained by at least k runs
    mask = attainment_counts >= k
    candidates = union[mask]
    
    if len(candidates) == 0:
        return AttainmentSurface(np.zeros((0, union.shape[1])), level=level)
    
    # Extract non-dominated subset of these candidates
    from ..core.run import Population
    pop = Population(candidates, candidates)
    # _calc_domination is now memory-efficient (chunked)
    indices = pop._calc_domination()
    
    return AttainmentSurface(candidates[~indices], level=level)

class AttainmentDiff(StatsResult):
    """
    Result of a comparison between two attainment surfaces.
    """
    def __init__(self, surf1, surf2, level):
        self.surf1 = surf1
        self.surf2 = surf2
        self.level = level
        
    def __iter__(self):
        # Allow unpacking: s1, s2 = attainment_diff(...)
        yield self.surf1
        yield self.surf2

    @property
    def volume_diff(self):
        """Difference in volume (S1 - S2). Positive means S1 is better."""
        return self.surf1.volume() - self.surf2.volume()

    def report(self) -> str:
        v1 = self.surf1.volume()
        v2 = self.surf2.volume()
        dv = v1 - v2
        better = self.surf1.name if dv > 0 else self.surf2.name
        
        lines = [
            f"--- Attainment Comparison Report (Level {self.level:.2f}) ---",
            f"  - {self.surf1.name} Volume: {v1:.6f}",
            f"  - {self.surf2.name} Volume: {v2:.6f}",
            f"  - Absolute Difference: {abs(dv):.6f}",
            f"\nDiagnosis: {better} dominates a larger objective region at this probability level."
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"<AttainmentDiff {self.surf1.name} vs {self.surf2.name}>"

def attainment_diff(exp1, exp2, level=0.5, workers=0):
    """
    Calculates the spatial difference in attainment between two experiments
    at a specific attainment level (default: 0.5/Median).
    
    Args:
        exp1, exp2: Experiment objects.
        level (float): Attainment level [0, 1].
        workers (int): [DEPRECATED] Parallel execution is no longer supported.
    
    Returns:
        AttainmentDiff object.
    """
    # Extract fronts here to avoid detached object graph overhead
    fronts1 = [np.array(run.front()) for run in exp1.runs]
    fronts2 = [np.array(run.front()) for run in exp2.runs]

    # Purely serial calculation
    surf1 = attainment(fronts1, level=level)
    surf2 = attainment(fronts2, level=level)

    surf1.name = exp1.name
    surf2.name = exp2.name
    
    return AttainmentDiff(surf1, surf2, level)

