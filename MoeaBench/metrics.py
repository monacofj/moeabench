# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from MoeaBench.GEN_hypervolume import GEN_hypervolume
from MoeaBench.GEN_igd import GEN_igd
from MoeaBench.GEN_gd import GEN_gd
from MoeaBench.GEN_gdplus import GEN_gdplus
from MoeaBench.GEN_igdplus import GEN_igdplus

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
         # Assuming exp is a MultiExperiment or Run
         # We need to access its LAST front or ALL fronts? 
         # analyse_metric_gen.normalize uses get_F_gen_non_dominate()[-1]
         if hasattr(exp, 'get_elements'): # Old API?
             # Wrapper for compatibility
             pass
         # New API: iterate runs
         for run in exp:
             all_fronts.append(run.front())
    
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
        return np.zeros(3), np.ones(3) # Fallback
        
    global_min = np.min(np.vstack(mins), axis=0)
    global_max = np.max(np.vstack(maxs), axis=0)
    
    return global_min, global_max


class MetricMatrix:
    """
    A matrix (Generations x Runs) of metric values.
    """
    def __init__(self, data, metric_name="Metric", source_name=None):
        self._data = np.array(data) # Shape (G, R) or (R, G)? 
        # Usually we want Trajectories as rows or columns?
        # API says:
        # hv.runs[i] -> Trajectory (all gens)
        # hv.gens[i] -> Distribution (all runs)
        
        # Let's say shape is (Runs, Gens) for storage, transposing for API if needed.
        # But API doc says: "Returns a matrix ... lines are gens, columns are runs"
        # So shape is (G, R).
        
        if self._data.ndim == 1:
             # Single run case
             self._data = self._data.reshape(-1, 1)
             
        self.metric_name = metric_name
        self.source_name = source_name

    def __repr__(self):
        return f"<MetricMatrix {self._data.shape} ({self.metric_name})>"

    def __array__(self):
        return self._data
        
    @property
    def runs(self):
        # runs[i] returns i-th column ? No, runs[i] usually means "Run i".
        # If matrix is G x R:
        # Run i is column i.
        return self._data.T # Returns (R, G), so [i] gets the row (Run i).

    @property
    def gens(self):
        # gens[i] returns row i.
        return self._data


def hypervolume(exp, ref=None):
    """
    Calculates Hypervolume for an experiment.
    Returns a MetricMatrix (G x R).
    """
    if ref is None: ref = []
    if not isinstance(ref, list): ref = [ref]
    
    # 1. Collect all data
    # We need F_GEN and F (final front) for all runs
    F_GENs = [] # List of lists of fronts
    Fs = []     # List of final fronts
    
    for run in exp:
        # Assuming run is our new Run object
        # We need raw internal access or API access
        # Run.py has _engine_result
        res = run._engine_result
        
        # Check if engine supports history
        if hasattr(res, 'get_F_GEN'):
             f_gen = res.get_F_GEN() # List of 2D arrays
             F_GENs.append(f_gen)
             
             # Final front (or non-dominated)
             # Use run.front() -> Non-Dominated
             # But hypervolume calculation usually uses ALL solutions or Non-Dominated?
             # GEN_hypervolume seems to take fgen.
             
             # For 'F' argument (reference for normalization), use run.front()
             Fs.append(run.front())
        else:
            # Fallback for engines without history?
            pass

    # 2. Normalize (Find Reference Point)
    min_val, max_val = normalize(ref, Fs)
    
    # 3. Calculate
    # result matrix: G x R
    # But runs might have different lengths?
    # Assume same length for matrix. If not, fill with NaN?
    
    max_gens = max(len(h) for h in F_GENs)
    n_runs = len(exp)
    
    mat = np.full((max_gens, n_runs), np.nan)
    
    for r_idx, (f_gen, f_last) in enumerate(zip(F_GENs, Fs)):
        # f_gen is list of fronts over time
        
        # Instantiating GEN_hypervolume for the WHOLE run? 
        # No, GEN_hypervolume seems to handle a list of fronts?
        # Let's check GEN_hypervolume signature in previous view_file.
        # It takes (fgen, dim, min_non, max_non).
        # And has .evaluate().
        
        # Lines 169 in IPL_MoeaBench:
        # return [GEN_hypervolume(fgen, f.shape[1], min_non, max_non) for fgen,f in zip(F_GEN, F)]
        
        # It seems F_GEN passed to set_hypervolume is a list of lists of fronts?
        # analyse_metric_gen line 24 creates F_GEN as list of runs, where each run is list of fronts.
        
        metric = GEN_hypervolume(f_gen, f_last.shape[1], min_val, max_val)
        values = metric.evaluate() # Returns list of floats
        
        # Fill matrix
        length = min(len(values), max_gens)
        mat[:length, r_idx] = values[:length]
        
    return MetricMatrix(mat, "Hypervolume", source_name=getattr(exp, 'name', None))

def get_reference_front(ref_exps, current_fronts):
    """
    Constructs the reference Pareto Front.
    If ref_exps is provided, extracts their fronts.
    Otherwise, uses the current fronts.
    Returns a non-dominated set of points.
    """
    all_fronts = []
    
    # Add external references
    for exp in ref_exps:
         if hasattr(exp, 'get_elements'): # Old API?
             pass 
         elif hasattr(exp, 'front'): # Single run/exp abstraction
             all_fronts.append(exp.front())
         elif hasattr(exp, '__iter__'): # MultiExperiment
             for run in exp:
                 all_fronts.append(run.front())
         else:
             # Assume array
             all_fronts.append(exp)
    
    # If no refs provided, usage strategy:
    # Hypervolume: uses min/max of current.
    # GD/IGD: needs a reference set. If none provided, commonly we use the aggregate current front.
    if not all_fronts and not ref_exps:
        all_fronts.extend(current_fronts)
        
    if not all_fronts:
        return None

    # Stack
    # Filter for empty
    valid = [f for f in all_fronts if len(f) > 0]
    if not valid: return None
    
    merged = np.vstack(valid)
    
    # Calculate non-dominated subset of merged
    # Simple N^2 or assuming small enough?
    # For correctness of GD/IGD, it should be non-dominated.
    # Let's assume we return all and let the metric handle it, OR filter it.
    # PyMoo metrics often expect the True PF.
    
    # Let's do a quick NDS filter if possible, or return merged.
    # Filter 
    is_dominated = np.zeros(merged.shape[0], dtype=bool)
    for i in range(len(merged)):
         curr = merged[i]
         # simple check
         if np.any(np.all(merged <= curr, axis=1) & np.any(merged < curr, axis=1)):
             is_dominated[i] = True
             
    return merged[~is_dominated]

def _calc_metric(exp, ref, MetricClass, name):
    if ref is None: ref = []
    if not isinstance(ref, list): ref = [ref]
    
    F_GENs = []
    Fs = []
    
    for run in exp:
        res = run._engine_result
        if hasattr(res, 'get_F_GEN'):
             f_gen = res.get_F_GEN()
             F_GENs.append(f_gen)
             Fs.append(run.front())

    # Helper for Hypervolume normalization
    min_val, max_val = normalize(ref, Fs)
    
    # Helper for GD/IGD reference front
    ref_front = get_reference_front(ref, Fs)
    
    max_gens = max(len(h) for h in F_GENs) if F_GENs else 0
    n_runs = len(exp)
    mat = np.full((max_gens, n_runs), np.nan)
    
    for r_idx, (f_gen, f_last) in enumerate(zip(F_GENs, Fs)):
        # Dispatch based on internal GEN_ class logic
        if name == "Hypervolume":
             metric = MetricClass(f_gen, f_last.shape[1], min_val, max_val)
        else:
             # GD, GD+, IGD, IGD+
             if ref_front is None:
                  # Cannot calculate without reference
                  values = np.full(len(f_gen), np.nan)
             else:
                  metric = MetricClass(f_gen, ref_front)
                  values = metric.evaluate()
        
        length = min(len(values), max_gens)
        mat[:length, r_idx] = values[:length]
        
    return MetricMatrix(mat, name, source_name=getattr(exp, 'name', None))

def gd(exp, ref=None):
    return _calc_metric(exp, ref, GEN_gd, "GD")

def gdplus(exp, ref=None):
    return _calc_metric(exp, ref, GEN_gdplus, "GD+")

def igd(exp, ref=None):
    return _calc_metric(exp, ref, GEN_igd, "IGD")

def igdplus(exp, ref=None):
    return _calc_metric(exp, ref, GEN_igdplus, "IGD+")

def plot_matrix(metric_matrices, mode='interactive'):
    """
    Plots a list of MetricMatrix objects.
    mode: 'interactive' (Plotly) or 'static' (Matplotlib)
    """
    if not isinstance(metric_matrices, (list, tuple)):
        metric_matrices = [metric_matrices]

    # Determine common name
    names = sorted(list(set(m.metric_name for m in metric_matrices)))
    if len(names) == 1:
        plot_name = names[0]
    else:
        plot_name = ", ".join(names)

    if mode == 'static':
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for mat in metric_matrices:
             data = mat.gens
             label = f"{mat.metric_name} ({mat.source_name})" if mat.source_name else mat.metric_name
             
             if data.shape[1] > 1:
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)
                gens = np.arange(1, len(mean) + 1)
                
                ax.plot(gens, mean, label=f'{label} Mean')
                # ax.fill_between(gens, mean-std, mean+std, alpha=0.3)
             else:
                ax.plot(np.arange(1, len(data)+1), data[:, 0], label=label)
        
        ax.set_title(f"{plot_name} over Time")
        ax.set_xlabel("Generation")
        ax.set_ylabel(plot_name)
        ax.legend()
        plt.show()
        
    else:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for mat in metric_matrices:
            # mat is MetricMatrix (G x R)
            # Calculate mean and std dev across runs (axis 1)
            data = mat.gens # (G, R)
            label = f"{mat.metric_name} ({mat.source_name})" if mat.source_name else mat.metric_name
            
            if data.shape[1] > 1:
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)
                gens = np.arange(1, len(mean) + 1)
                
                # Plot Mean
                fig.add_trace(go.Scatter(
                    x=gens, y=mean,
                    mode='lines',
                    name=f'{label} Mean'
                ))
                
                # TODO: Add shadow/band for std dev
                
            else:
                # Single run
                fig.add_trace(go.Scatter(
                    x=np.arange(1, len(data)+1),
                    y=data[:, 0],
                    mode='lines',
                    name=label
                ))
                
        fig.update_layout(title=f"{plot_name} over Time", xaxis_title="Generation", yaxis_title=plot_name)
        # In non-interactive environments (like verifying script), verify it doesn't block or require browser if using static?
        # Plotly .show() opens browser.
        # Matplotlib .show() opens window.
        fig.show()

# TODO: Add IGD, GD, etc. same pattern.
