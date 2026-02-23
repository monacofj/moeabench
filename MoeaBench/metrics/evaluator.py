# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import logging
from typing import Optional, Union, Any, List
from .GEN_hypervolume import GEN_hypervolume
from .GEN_mc_hypervolume import GEN_mc_hypervolume
from .GEN_igd import GEN_igd
from .GEN_gd import GEN_gd
from .GEN_gdplus import GEN_gdplus
from .GEN_igdplus import GEN_igdplus
from ..core.base import Reportable
from ..core.run import Population
from ..defaults import defaults
import warnings

# We need the helper logic from analyse to calculate reference points
# Since analyse is a mixin class, we might need to extract 'normalize' to a util or duplicate it.
# To avoid complex inheritance, let's keep it simple here.

def normalize(ref_exps, all_current_objs_list):
    """
    Calculates the global min/max objective values across reference experiments 
    and the current experiment to establish the bounding box for Hypervolume.
    """
    # Collect all fronts from references
    all_fronts = []
    
    # Add external references
    for exp in ref_exps:
        if hasattr(exp, 'runs'):
            # It's an experiment
            for run in exp:
                all_fronts.append(run.history('nd')[-1]) # Use last gen
        elif isinstance(exp, np.ndarray):
            # It's a raw array
            all_fronts.append(exp)
        elif hasattr(exp, 'front') and callable(exp.front):
             all_fronts.append(exp.front())
             
    # Add current experiment fronts (from all_current_objs_list)
    # F is a list of arrays (one per run)
    for f in all_current_objs_list:
        all_fronts.append(f)
        
    if not all_fronts:
        return None, None

    # Stack and find min/max
    # Note: fronts might have different sizes
    mins = []
    maxs = []
    for f in all_fronts:
        if len(f) > 0:
            mins.append(np.min(f, axis=0))
            maxs.append(np.max(f, axis=0))
            
    if not mins:
        # Fallback to zeros/ones but with correct dimensionality M
        # We try to infer M from current fronts if possible
        M = 0
        for f in all_current_objs_list:
            if len(f) > 0:
                M = f.shape[1]
                break
        if M == 0:
            M = 3 # Absolute default if nothing is found
        return np.zeros(M), np.ones(M)
        
    global_min = np.min(np.vstack(mins), axis=0)
    global_max = np.max(np.vstack(maxs), axis=0)
    
    return global_min, global_max


class MetricMatrix(Reportable):
    """
    A matrix (Generations x Runs) of metric values.
    """
    def __init__(self, data, metric_name="Metric", source_name=None):
        self._data = np.array(data) 
        
        # Internal Storage Policy: (Generations, Runs)
        # We ensure it's always 2D
        if self._data.ndim == 1:
             # Case: Array of generations for a single run
             self._data = self._data.reshape(-1, 1)
             
        self.metric_name = metric_name
        self.source_name = source_name
        
        # Determine if this metric is scaled 
        self.is_ratio = "(Ratio)" in metric_name or "(Rel)" in metric_name
        self.is_raw = "(Raw)" in metric_name
        self.is_abs = "(Abs)" in metric_name

    def report(self, **kwargs) -> str:
        """Narrative report of the metric performance and stability."""
        use_md = kwargs.get('markdown', False)
        data = self._data
        if data.size == 0:
            if use_md:
                return f"### Metric Report: {self.metric_name}\n**Status**: No data available"
            return f"--- Metric Report: {self.metric_name} ---\n  Status: No data available"

        # Distribution at the last generation
        final_dist = data[-1, :]
        valid_final = final_dist[np.isfinite(final_dist)]
        
        if len(valid_final) == 0:
            if use_md:
                return f"### Metric Report: {self.metric_name}\n**Status**: All values are NaN"
            return f"--- Metric Report: {self.metric_name} ---\n  Status: All values are NaN"

        mean = np.mean(valid_final)
        std = np.std(valid_final)
        best = np.max(valid_final) # Assuming higher is better (Hypervolume)
        if any(m in self.metric_name.lower() for m in ['igd', 'gd', 'spacing']):
            best = np.min(valid_final)

        cv = std / (abs(mean) + 1e-9)
        prec = defaults.precision
        source_info = f" ({self.source_name})" if self.source_name else ""

        if cv < defaults.cv_tolerance:
            stability = f"High (CV={cv:.{prec}f} < {defaults.cv_tolerance})"
        elif cv > defaults.cv_moderate:
            stability = f"Low (CV={cv:.{prec}f} > {defaults.cv_moderate})"
        else:
            stability = f"Moderate ({defaults.cv_tolerance} <= CV={cv:.{prec}f} <= {defaults.cv_moderate})"

        if use_md:
            lines = [
                f"### Metric Report: {self.metric_name}{source_info}",
            ]
            
            if "Hypervolume" in self.metric_name:
                if self.is_ratio:
                    lines.extend([
                        "> [!IMPORTANT]",
                        "> **Competitive Efficiency**: What percentage of the best observed performance did this algorithm achieve?",
                        "> *Note: Values are scaled by the maximum session volume ($1.0$ ceiling).* ",
                        ""
                    ])
                elif self.is_raw:
                    lines.extend([
                        "> [!NOTE]",
                        "> **Physical Objective Space**: How much objective space has been physically conquered within the global search boundaries established in this session?",
                        ""
                    ])
                elif self.is_abs:
                    lines.extend([
                        "> [!TIP]",
                        "> **Theoretical Optimality**: How close is this algorithm to mathematical perfection?",
                        "> *Note: Values are normalized by the pre-calculated Ground Truth of the problem ($1.0$ = Opt).* ",
                        ""
                    ])

            lines.extend([
                "#### Final Performance (Last Gen)",
                f"- **Mean**: {mean:.{prec}f}",
                f"- **StdDev**: {std:.{prec}f}",
                f"- **Best**: {best:.{prec}f}",
                "",
                "#### Search Dynamics",
                f"- **Runs**: {data.shape[1]}",
                f"- **Generations**: {data.shape[0]}",
                f"- **Stability**: {stability}"
            ])
        else:
            lines = [
                f"--- Metric Report: {self.metric_name}{source_info} ---"
            ]
            
            if "Hypervolume" in self.metric_name:
                if self.is_ratio:
                    lines.extend([
                        "  Question: What is the competitive efficiency relative to best session performance?"
                    ])
                elif self.is_raw:
                    lines.extend([
                        "  Question: How much objective space has been physically conquered?"
                    ])
            
            lines.extend([
                f"  Final Performance (Last Gen):",
                f"    - Mean: {mean:.{prec}f}",
                f"    - StdDev: {std:.{prec}f}",
                f"    - Best: {best:.{prec}f}",
                f"  Search Dynamics:",
                f"    - Runs: {data.shape[1]}",
                f"    - Generations: {data.shape[0]}",
                f"    - Stability: {stability}"
            ])
        
        return "\n".join(lines)

    def __getitem__(self, key: Union[int, slice]) -> 'MetricMatrix':
        """
        Selectors: Consistent with Experiment indexing (by Run).
        Selects columns (Runs) from the Generations x Runs matrix.
        """
        # _data is (G, R), key slices R (axis 1)
        if isinstance(key, int):
            # Preserve 2D shape (G, 1) to remain a MetricMatrix object
            new_data = self._data[:, key:key+1]
        else:
            new_data = self._data[:, key]
            
        return MetricMatrix(new_data, self.metric_name, self.source_name)

    def __len__(self):
        """Returns the number of runs (consistent with Experiment)."""
        return self._data.shape[1]

    def __repr__(self):
        if self._data.size == 1:
            return f"{self._data.item():.6f}"
        return super().__repr__()

    def __float__(self):
        if self._data.size == 1:
            return float(self._data.item())
        raise TypeError(f"MetricMatrix ({self.metric_name}) contains {self._data.size} values and cannot be converted to a single float.")

    def __format__(self, format_spec):
        if self._data.size == 1:
            return format(float(self._data.item()), format_spec)
        return format(str(self), format_spec)

    def __array__(self):
        return self._data
        
    def run(self, i=-1):
        """
        Returns the metric trajectory (all generations) for a specific run.
        defaults to the last run (-1).
        """
        if self._data.ndim == 1:
            return self._data
        return self._data[:, i]

    def gen(self, n=-1):
        """
        Returns the metric distribution (all runs) for a specific generation.
        Defaults to the last generation (-1).
        """
        return self._data[n, :]

    # Legacy Aliases
    def runs(self, idx=-1): return self.run(idx)
    def gens(self, idx=-1): return self.gen(idx)
        
    @property
    def values(self):
        """Returns the raw numpy array (Generations x Runs)."""
        return self._data


    @property
    def last(self):
        """Shortcut for the mean value of the final generation."""
        return self.mean()

    def mean(self, n=-1):
        """Returns the mean value of the metric at generation n."""
        dist = self.gen(n)
        return float(np.mean(dist[np.isfinite(dist)]))

    def std(self, n=-1):
        """Returns the standard deviation of the metric at generation n."""
        dist = self.gen(n)
        return float(np.std(dist[np.isfinite(dist)]))

    def best(self, n=-1):
        """Returns the best value of the metric at generation n (handles min/max logic)."""
        dist = self.gen(n)
        valid = dist[np.isfinite(dist)]
        if not len(valid): return np.nan
        
        if any(m in self.metric_name.lower() for m in ['igd', 'gd', 'spacing']):
            return float(np.min(valid))
        return float(np.max(valid))


def _extract_data(data, gens: Optional[Union[int, slice]] = None):
    """
    Refines input into 
    (List[RunHistories], List[FinalFronts], SourceName, NumRuns)
    """
    from ..core.run import Run, Population
    from ..core.experiment import experiment

    # If gens is int, treat as slice[:gens]
    if gens is not None and isinstance(gens, int):
        if gens == -1:
            gens = slice(-1, None)
        else:
            gens = slice(gens)

    if isinstance(data, experiment):
        histories = [r.history('nd') for r in data]
        if gens is not None:
             histories = [h[gens] for h in histories]
        return histories, [r.front() for r in data], data.name, len(data)
    
    if isinstance(data, Run):
        h = data.history('nd')
        if gens is not None: h = h[gens]
        return [h], [data.front()], data.name, 1
        
    if isinstance(data, Population):
        return [[data.objectives]], [data.objectives], data.label, 1
        
    if isinstance(data, np.ndarray):
        # Treat as a single population
        return [[data]], [data], "Array", 1

    # Fallback for generic iterables
    try:
        histories = []
        fronts = []
        for item in data:
            if hasattr(item, 'history'): # Run-like
                h = item.history('nd')
                if gens is not None:
                    h = h[gens]
                    if isinstance(h, np.ndarray) and h.ndim == 1: h = [h] # single pop case
                histories.append(h)
                fronts.append(item.front())
            else:
                histories.append([item])
                fronts.append(item)
        
        # Adjust histories if not Run-like but gens provided for top-level list
        if gens is not None and not hasattr(data, 'history'):
            histories = histories[gens]
            fronts = fronts[gens]

        return histories, fronts, None, len(histories)
    except:
        raise TypeError(f"Unsupported data type for metric calculation: {type(data)}")

def hypervolume(exp, ref=None, mode='auto', scale='raw', n_samples=100000, gens=None, joint=True):
    """
    Calculates Hypervolume for an experiment, run, or population.
    Returns a MetricMatrix (G x R).

    Args:
        exp: Experiment, Run, or Population object.
        ref: Reference set/experiment for normalization bounding box.
        mode (str): Algorithm to use: 'auto' (default), 'exact', or 'fast'.
        scale (str): Scaling perspective: 'raw' (default), 'relative', or 'absolute'.
        n_samples (int): Number of Monte Carlo samples for 'fast'/'auto' mode.
        gens (int or slice): Limit calculation to specific generation(s).
        joint (bool): If True (default), uses the union of 'exp' and 'ref' to establish 
                     the bounding box. If False, ignores 'ref' for normalization, 
                     providing an independent (self-referenced) perspective.
    """
    if ref is None: ref = []
    if not isinstance(ref, list): ref = [ref]
    
    # Contextual reference: If joint=False, we ignore external references for normalization
    effective_ref = ref if joint else []
    
    # --- 0. MOP Validation & Meta-data ---
    # Check if we are mixing different problems
    scale = str(scale).lower()
    from ..diagnostics.baselines import load_offline_baselines
    
    mop_names = []
    for item in [exp] + ref:
        if hasattr(item, 'mop'):
            mop_names.append(getattr(item.mop, 'name', item.mop.__class__.__name__))
        elif hasattr(item, 'evaluation') and hasattr(item, 'pf'):
            # It's likely a MOP object
            mop_names.append(getattr(item, 'name', item.__class__.__name__))
            
    if len(set(mop_names)) > 1:
        msg = f"Hypervolume: Mixed MOPs detected in session: {list(set(mop_names))}. " \
              f"Comparing different problems yields invalid geometric results."
        if scale == 'absolute':
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    
    # 1. Collect all data
    F_GENs, Fs, name, n_runs = _extract_data(exp, gens=gens)

    # 2. Normalize (Find Reference Point)
    min_val, max_val = normalize(effective_ref, Fs)
    
    # 3. Calculate
    max_gens = max(len(h) for h in F_GENs) if F_GENs else 0
    mat = np.full((max_gens, n_runs), np.nan)
    
    for r_idx, (f_gen, f_last) in enumerate(zip(F_GENs, Fs)):
        M = f_last.shape[1] if len(f_last) > 0 else 0
        if M == 0 and len(f_gen) > 0:
             for f in f_gen:
                  if len(f) > 0:
                       M = f.shape[1]
                       break
        
        if M > 0:
            # Smart Selection of Algorithm
            use_mc = False
            if mode == 'fast':
                use_mc = True
            elif mode == 'auto' and M > 6:
                use_mc = True
                logging.info(f"Hypervolume: High-dimensional space (M={M}) detected. "
                             f"Switching to Monte Carlo approximation (n={n_samples}).")
            elif mode == 'exact' and M > 8:
                warnings.warn(f"Exact Hypervolume calculation for M={M} objectives has exponential complexity O(2^M) "
                              f"and may be extremely slow. Consider using mode='auto' or mode='fast'.")
            
            if use_mc:
                metric = GEN_mc_hypervolume(f_gen, M, min_val, max_val, n_samples=n_samples)
            else:
                metric = GEN_hypervolume(f_gen, M, min_val, max_val)
                
            values = metric.evaluate()
            
            # Fill matrix
            length = min(len(values), max_gens)
            mat[:length, r_idx] = values[:length]
        
    # --- 4. Dynamic Benchmarking ---
    # We normalize all absolute HVs by a reference HV to establish a 1.0 ceiling
    # --- 4. Scale Post-Processing ---
    scale = str(scale).lower()
    if scale in ['relative', 'ratio']:
        if scale == 'ratio':
            warnings.warn("scale='ratio' is deprecated, use scale='relative' instead.", DeprecationWarning)
            
        if effective_ref:
            # A) Explicit Reference (e.g., Competition Mode)
            ref_list = effective_ref
            ref_hvs = []
            for r in ref_list:
                _, r_fs, _, _ = _extract_data(r)
                for f in r_fs:
                    if len(f) > 0:
                        if mode == 'fast' or (mode == 'auto' and M > 6):
                            m = GEN_mc_hypervolume([f], M, min_val, max_val, n_samples=n_samples)
                        else:
                            m = GEN_hypervolume([f], M, min_val, max_val)
                        ref_hvs.append(float(m.evaluate()[0]))
            
            ref_hv_val = np.max(ref_hvs) if ref_hvs else 0
        else:
            # B) Implicit Self-Reference (Best run in current data)
            ref_hv_val = np.nanmax(mat[-1, :]) if mat.size > 0 else 0

        if ref_hv_val > 0:
            mat /= ref_hv_val
            
        final_name = "Hypervolume (Relative)"
    
    elif scale == 'absolute':
        # Retrieve Ground Truth from Calibration Registry
        mop_obj = getattr(exp, 'mop', None)
        if mop_obj is None:
             raise ValueError("Hypervolume 'absolute' scale requires an experiment with an associated MOP.")
             
        mop_id = getattr(mop_obj, 'name', mop_obj.__class__.__name__)
        try:
            bases = load_offline_baselines()
            gt_registry = bases.get("_gt_registry", {})
            if mop_id not in gt_registry:
                raise ValueError(f"MOP '{mop_id}' is not calibrated. Run mop.calibrate() first to enable absolute scale.")
            
            gt = np.array(gt_registry[mop_id])
            
            # Calculate Reference HV (1.0 Ceiling) using the GT
            if mode == 'fast' or (mode == 'auto' and M > 6):
                m = GEN_mc_hypervolume([gt], M, min_val, max_val, n_samples=n_samples)
            else:
                m = GEN_hypervolume([gt], M, min_val, max_val)
            
            gt_hv = float(m.evaluate()[0])
            if gt_hv > 0:
                mat /= gt_hv
            
        except Exception as e:
            if isinstance(e, ValueError): raise e
            raise RuntimeError(f"Failed to calculate absolute hypervolume: {e}")
            
        final_name = "Hypervolume (Absolute)"

    elif scale == 'raw':
        final_name = "Hypervolume (Raw)"
    else:
        raise ValueError(f"Unknown scale parameter: {scale}. Use 'raw', 'relative', or 'absolute'.")

    return MetricMatrix(mat, final_name, source_name=name)

def get_reference_front(ref_exps, current_fronts):
    """
    Constructs the reference Pareto Front.
    """
    all_fronts = []
    
    # Add external references
    for ref in ref_exps:
         _, fronts, _, _ = _extract_data(ref)
         all_fronts.extend(fronts)
    
    # If no refs provided, usage strategy:
    if not all_fronts and not ref_exps:
        all_fronts.extend(current_fronts)
        
    if not all_fronts:
        return None

    # Filter for empty
    valid = [f for f in all_fronts if len(f) > 0]
    if not valid: return None
    
    merged = np.vstack(valid)
    
    # Simple NDS filter
    is_dominated = np.zeros(merged.shape[0], dtype=bool)
    for i in range(len(merged)):
         curr = merged[i]
         # check if any other point dominates curr
         if np.any(np.all(merged <= curr, axis=1) & np.any(merged < curr, axis=1)):
             is_dominated[i] = True
             
    return merged[~is_dominated]

def _calc_metric(exp, ref, MetricClass, name, gens=None):
    if ref is None: ref = []
    if not isinstance(ref, list): ref = [ref]
    
    F_GENs, Fs, source_name, n_runs = _extract_data(exp, gens=gens)

    # Helper for Hypervolume normalization
    min_val, max_val = normalize(ref, Fs)
    
    # Helper for GD/IGD reference front
    ref_front = get_reference_front(ref, Fs)
    
    max_gens = max(len(h) for h in F_GENs) if F_GENs else 0
    mat = np.full((max_gens, n_runs), np.nan)
    
    for r_idx, (f_gen, f_last) in enumerate(zip(F_GENs, Fs)):
        # Dispatch based on internal GEN_ class logic
        if name == "Hypervolume":
             M = f_last.shape[1] if len(f_last) > 0 else 0
             if M > 0:
                 metric = MetricClass(f_gen, M, min_val, max_val)
                 values = metric.evaluate()
             else:
                 values = np.full(len(f_gen), np.nan)
        else:
             # GD, GD+, IGD, IGD+
             if ref_front is None:
                  values = np.full(len(f_gen), np.nan)
             else:
                  metric = MetricClass(f_gen, ref_front)
                  values = metric.evaluate()
        
        length = min(len(values), max_gens)
        mat[:length, r_idx] = values[:length]
        
    return MetricMatrix(mat, name, source_name=source_name)

def gd(exp, ref=None, gens=None):
    if ref is None and hasattr(exp, 'optimal_front'):
        try:
            ref = exp.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"GD: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    
    # Check if single population (ndarray)
    if isinstance(exp, np.ndarray):
        if ref is None: return MetricMatrix(np.array([np.nan]))
        from .GEN_gd import GEN_gd
        # GEN_gd expects (Hist, Ref)
        metric = GEN_gd([exp], ref)
        return MetricMatrix(metric.evaluate(), "GD")

    from .GEN_gd import GEN_gd
    return _calc_metric(exp, ref, GEN_gd, "GD", gens=gens)

def gdplus(exp, ref=None, gens=None):
    if ref is None and hasattr(exp, 'optimal_front'):
        try:
            ref = exp.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"GD+: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    
    if isinstance(exp, np.ndarray):
        if ref is None: return MetricMatrix(np.array([np.nan]))
        from .GEN_gdplus import GEN_gdplus
        metric = GEN_gdplus([exp], ref)
        return MetricMatrix(metric.evaluate(), "GD+")

    from .GEN_gdplus import GEN_gdplus
    return _calc_metric(exp, ref, GEN_gdplus, "GD+", gens=gens)

def igd(exp, ref=None, gens=None):
    if ref is None and hasattr(exp, 'optimal_front'):
        try:
            ref = exp.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"IGD: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    
    if isinstance(exp, np.ndarray):
        if ref is None: return MetricMatrix(np.array([np.nan]))
        from .GEN_igd import GEN_igd
        metric = GEN_igd([exp], ref)
        return MetricMatrix(metric.evaluate(), "IGD")

    from .GEN_igd import GEN_igd
    return _calc_metric(exp, ref, GEN_igd, "IGD", gens=gens)

def emd(exp, ref=None, gens=None):
    """
    Computes the Earth Mover's Distance (Wasserstein) between population and reference.
    For multivariate data, this implementation uses the average 1D Wasserstein distance 
    per objective as a fast and robust distributional shift proxy.
    """
    if ref is None and hasattr(exp, 'optimal_front'):
        try:
            ref = exp.optimal_front()
        except:
            pass
    
    from scipy.stats import wasserstein_distance
    
    def _calc_emd_pair(pts, r_pts):
        if pts is None or r_pts is None or len(pts) == 0 or len(r_pts) == 0:
            return np.nan
        M = pts.shape[1]
        w_dists = []
        for m in range(M):
            d = wasserstein_distance(pts[:, m], r_pts[:, m])
            w_dists.append(d)
        
        # DEBUG
        # if np.mean(w_dists) > 0.01:
        #    print(f"DEBUG EMD: M={M}, Dists={w_dists}")
        #    print(f"PTS Shape: {pts.shape}, REF Shape: {r_pts.shape}")
        return np.mean(w_dists)

    F_GENs, Fs, source_name, n_runs = _extract_data(exp, gen=gen)
    ref_front = get_reference_front(ref, Fs)
    
    # DEBUG
    print(f"DEBUG: ref_front shape: {ref_front.shape if ref_front is not None else 'None'}")
    
    if ref_front is None:
        return MetricMatrix(np.full((1, n_runs), np.nan), "EMD")

    max_gens = max(len(h) for h in F_GENs) if F_GENs else 0
    mat = np.full((max_gens, n_runs), np.nan)
    
    for r_idx, f_gen in enumerate(F_GENs):
        values = []
        for g_pop in f_gen:
            values.append(_calc_emd_pair(g_pop, ref_front))
        
        length = min(len(values), max_gens)
        mat[:length, r_idx] = values[:length]
        
    return MetricMatrix(mat, "EMD", source_name=source_name)

def igdplus(exp, ref=None, gens=None):
    if ref is None and hasattr(exp, 'optimal_front'):
        try:
            ref = exp.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"IGD+: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    
    if isinstance(exp, np.ndarray):
        if ref is None: return MetricMatrix(np.array([np.nan]))
        from .GEN_igdplus import GEN_igdplus
        metric = GEN_igdplus([exp], ref)
        return MetricMatrix(metric.evaluate(), "IGD+")

    from .GEN_igdplus import GEN_igdplus
    return _calc_metric(exp, ref, GEN_igdplus, "IGD+", gens=gens)

def plot_matrix(metric_matrices, mode='auto', show_bounds=False, title=None, **kwargs):
    """
    Plots a list of MetricMatrix objects.
    mode: 'auto' (detects environment), 'interactive' (Plotly) or 'static' (Matplotlib)
    """
    # Environment detection for 'auto' mode
    if mode == 'auto':
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                mode = 'interactive'
            else:
                mode = 'static'
        except (ImportError, NameError):
            mode = 'static'
            
    if not isinstance(metric_matrices, (list, tuple)):
        metric_matrices = [metric_matrices]

    # Handle nested tuples/lists from wrappers like timeplot(*args)
    if len(metric_matrices) == 1 and isinstance(metric_matrices[0], (list, tuple)):
        metric_matrices = metric_matrices[0]

    # Determine common name
    names = sorted(list(set(m.metric_name for m in metric_matrices)))
    if len(names) == 1:
        plot_name = names[0]
    else:
        plot_name = ", ".join(names)

    final_title = title if title else f"{plot_name} over Time"

    if mode == 'static':
        import matplotlib.pyplot as plt
        
        ax = kwargs.get('ax', None)
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        else:
            fig = ax.get_figure()
        
        lstyles = kwargs.get('linestyles', ['-', '--', ':', '-.'])
        if not isinstance(lstyles, (list, tuple)): lstyles = [lstyles]

        labels = kwargs.get('labels', [])
        for i, mat in enumerate(metric_matrices):
             data = mat.values
             
             if i < len(labels):
                 label = labels[i]
             else:
                 # Standard Legend Logic: Name (G: XX, R: YY)
                 name = mat.source_name if mat.source_name else mat.metric_name
                 G, R = data.shape
                 meta = []
                 
                 # Rule: If G=1 (snapshot), specify G. If history (G>1), omit G.
                 if G == 1:
                     meta.append(f"G: 1")
                 
                 # Rule: If R=1 (single run), specify R. If multiple (aggregated), R is implied 
                 # unless it's a specific subset (not detectable here yet, so we omit if R > 1).
                 if R == 1:
                     meta.append(f"R: 1")
                     
                 suffix = f" ({', '.join(meta)})" if meta else ""
                 label = f"{name}{suffix}"
             
             ls = lstyles[i % len(lstyles)]
             
             if data.shape[1] > 1:
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)
                gens = np.arange(1, len(mean) + 1)
                v_min = np.nanmin(data, axis=1)
                v_max = np.nanmax(data, axis=1)
                
                ax.plot(gens, mean, label=label, linestyle=ls)
                ax.fill_between(gens, np.maximum(0, mean-std), mean+std, alpha=0.2)
                
                if show_bounds:
                    ax.plot(gens, v_min, '--', color=ax.get_lines()[-1].get_color(), alpha=0.5, linewidth=1)
                    ax.plot(gens, v_max, '--', color=ax.get_lines()[-1].get_color(), alpha=0.5, linewidth=1)
             else:
                ax.plot(np.arange(1, len(data)+1), data[:, 0], label=label, linestyle=ls)
        
        ax.set_title(final_title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(plot_name)
        ax.legend()
        if kwargs.get('show', True) and kwargs.get('ax') is None:
            plt.show()
        
        return fig, ax
        
    else:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for mat in metric_matrices:
            data = mat.values
            label = f"{mat.metric_name} ({mat.source_name})" if mat.source_name else mat.metric_name
            
            if data.shape[1] > 1:
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)
                gens = np.arange(1, len(mean) + 1)
                v_min = np.nanmin(data, axis=1)
                v_max = np.nanmax(data, axis=1)
                
                fig.add_trace(go.Scatter(
                    x=gens, y=mean,
                    mode='lines',
                    name=label,
                    line=dict(width=3),
                    hovertemplate=f"{label}<br>Gen: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                ))
                
                lower_bound = np.maximum(0, mean - std)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([gens, gens[::-1]]),
                    y=np.concatenate([mean + std, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(100, 100, 100, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f'{label} StdDev'
                ))
                
                if show_bounds:
                    fig.add_trace(go.Scatter(
                        x=gens, y=v_min,
                        mode='lines',
                        line=dict(dash='dash', width=1),
                        name=f'{label} Min',
                        showlegend=False,
                        opacity=0.5,
                        hovertemplate=f"{label} Min<br>Gen: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                    ))
                    fig.add_trace(go.Scatter(
                        x=gens, y=v_max,
                        mode='lines',
                        line=dict(dash='dash', width=1),
                        name=f'{label} Max',
                        showlegend=False,
                        opacity=0.5,
                        hovertemplate=f"{label} Max<br>Gen: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                    ))
                
            else:
                fig.add_trace(go.Scatter(
                    x=np.arange(1, len(data)+1),
                    y=data[:, 0],
                    mode='lines',
                    name=label,
                    hovertemplate=f"{label}<br>Gen: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                ))
                
        fig.update_layout(
            title=final_title, 
            xaxis_title="Generation", 
            yaxis_title=plot_name,
            hovermode='closest'
        )
        if kwargs.get('show', True):
            fig.show()

def front_size(exp, mode='run', gens=None):
    """
    Calculates the proportion of non-dominated individuals (Front Size)
    relative to the total population size at each generation.
    Returns a MetricMatrix (G x R) or (G x 1) if mode='consensus'.

    Args:
        exp: Experiment, Run, or Population object.
        mode (str): 'run' (per-run ratio) or 'consensus' (superfront density).
        gens (int or slice): Limit calculation to specific generation(s).
    """
    # 1. Data extraction logic
    from ..core.run import Run, Population
    from ..core.experiment import experiment
    from ..stats.stratification import strata
    
    if isinstance(exp, experiment):
        histories = [r.history('f') for r in exp._runs]
        name = exp.name
    elif isinstance(exp, Run):
        histories = [exp.history('f')]
        name = exp.name
    elif isinstance(exp, Population):
        histories = [[exp.objectives]]
        name = exp.label
    elif isinstance(exp, np.ndarray):
        histories = [[exp]]
        name = "Array"
    else:
        from .evaluator import _extract_data
        histories, _, name, _ = _extract_data(exp, gens=gens)

    # 2. Slice generations if requested
    if gens is not None and not isinstance(exp, Population):
        g_slice = slice(-1, None) if gens == -1 else (gens if isinstance(gens, slice) else slice(gens))
        histories = [h[g_slice] for h in histories]

    n_runs = len(histories)
    max_gens = max(len(h) for h in histories) if histories else 0
    mode = str(mode).lower()
    
    if mode == 'consensus':
        # AGGREGATE MODE: (G x 1) matrix
        mat = np.full((max_gens, 1), np.nan)
        for g_idx in range(max_gens):
            pops_at_g = []
            for r_h in histories:
                if g_idx < len(r_h) and r_h[g_idx] is not None:
                    pops_at_g.append(r_h[g_idx])
            
            if not pops_at_g:
                continue
                
            combined_objs = np.vstack(pops_at_g)
            try:
                # Calculate non-dominance ratio on the combined cloud
                s_res = strata(Population(combined_objs))
                n_nd = np.sum(s_res.rank_array == 1)
                mat[g_idx, 0] = n_nd / len(combined_objs)
            except Exception:
                mat[g_idx, 0] = 1.0 # Fallback
                
        label = f"Front Size (Consensus)"
    else:
        # PER-RUN MODE: (G x R) matrix
        mat = np.full((max_gens, n_runs), np.nan)
        for r_idx, h_run in enumerate(histories):
            for g_idx, pop_data in enumerate(h_run):
                if pop_data is None or len(pop_data) == 0:
                    mat[g_idx, r_idx] = 0.0
                    continue
                
                try:
                    active_pop = Population(pop_data) if isinstance(pop_data, np.ndarray) else pop_data
                    s_res = strata(active_pop)
                    n_nd = np.sum(s_res.rank_array == 1)
                    n_tot = len(pop_data)
                    mat[g_idx, r_idx] = n_nd / n_tot
                except Exception:
                    mat[g_idx, r_idx] = 1.0
        
        label = f"Front Size (Ratio)"

    return MetricMatrix(mat, metric_name=label, source_name=name)

# Alias for convenience
nd_ratio = front_size

# TODO: Add IGD, GD, etc. same pattern.
