# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import logging
from .GEN_hypervolume import GEN_hypervolume
from .GEN_mc_hypervolume import GEN_mc_hypervolume
from .GEN_igd import GEN_igd
from .GEN_gd import GEN_gd
from .GEN_gdplus import GEN_gdplus
from .GEN_igdplus import GEN_igdplus
from ..core.base import Reportable
from ..defaults import defaults

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

    def report(self, **kwargs) -> str:
        """Narrative report of the metric performance and stability."""
        data = self._data
        if data.size == 0:
            return f"--- Metric Report: {self.metric_name} ---\n  Status: No data available"

        # Distribution at the last generation
        # data is (G, R), so data[-1, :] is the final distribution
        final_dist = data[-1, :]
        valid_final = final_dist[np.isfinite(final_dist)]
        
        if len(valid_final) == 0:
            return f"--- Metric Report: {self.metric_name} ---\n  Status: All values are NaN"

        mean = np.mean(valid_final)
        std = np.std(valid_final)
        best = np.max(valid_final) # Assuming higher is better (Hypervolume)
        # Check if it might be IGD/GD where lower is better
        if any(m in self.metric_name.lower() for m in ['igd', 'gd', 'spacing']):
            best = np.min(valid_final)

        cv = std / (abs(mean) + 1e-9)
        prec = defaults.precision

        if cv < defaults.cv_tolerance:
            stability = f"High (CV={cv:.{prec}f} < {defaults.cv_tolerance})"
        elif cv > defaults.cv_moderate:
            stability = f"Low (CV={cv:.{prec}f} > {defaults.cv_moderate})"
        else:
            stability = f"Moderate ({defaults.cv_tolerance} <= CV={cv:.{prec}f} <= {defaults.cv_moderate})"

        source_info = f" ({self.source_name})" if self.source_name else ""
        
        lines = [
            f"--- Metric Report: {self.metric_name}{source_info} ---",
            f"  Final Performance (Last Gen):",
            f"    - Mean: {mean:.{prec}f}",
            f"    - StdDev: {std:.{prec}f}",
            f"    - Best: {best:.{prec}f}",
            f"  Search Dynamics:",
            f"    - Runs: {data.shape[1]}",
            f"    - Generations: {data.shape[0]}",
            f"    - Stability: {stability}"
        ]
        
        return "\n".join(lines)

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
        
    def runs(self, idx=-1):
        """
        Returns the metric trajectory (all generations) for a specific run.
        defaults to the last run (-1).
        
        Args:
            idx (int): Run index.
            
        Returns:
            np.ndarray: 1D array of metric values over time.
        """
        if self._data.ndim == 1:
            return self._data
        return self._data[:, idx]

    def gens(self, idx=-1):
        """
        Returns the metric distribution (all runs) for a specific generation.
        Defaults to the last generation (-1).
        
        Args:
            idx (int): Generation index.
            
        Returns:
            np.ndarray: 1D array of metric values across runs.
        """
        return self._data[idx, :]
        
    @property
    def values(self):
        """Returns the raw numpy array (Generations x Runs)."""
        return self._data


def _extract_data(data):
    """
    Refines input into 
    (List[RunHistories], List[FinalFronts], SourceName, NumRuns)
    """
    from ..core.run import Run, Population
    from ..core.experiment import experiment

    if isinstance(data, experiment):
        return [r.history('nd') for r in data], [r.front() for r in data], data.name, len(data)
    
    if isinstance(data, Run):
        return [data.history('nd')], [data.front()], data.name, 1
        
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
                histories.append(item.history('nd'))
                fronts.append(item.front())
            else:
                histories.append([item])
                fronts.append(item)
        return histories, fronts, None, len(histories)
    except:
        raise TypeError(f"Unsupported data type for metric calculation: {type(data)}")

def hypervolume(exp, ref=None, mode='auto', n_samples=100000):
    """
    Calculates Hypervolume for an experiment, run, or population.
    Returns a MetricMatrix (G x R).

    Args:
        exp: Experiment, Run, or Population object.
        ref: Reference set/experiment for normalization.
        mode (str): 'auto' (default), 'exact', or 'fast'.
        n_samples (int): Number of Monte Carlo samples for 'fast'/'auto' mode.
    """
    if ref is None: ref = []
    if not isinstance(ref, list): ref = [ref]
    
    # 1. Collect all data
    F_GENs, Fs, name, n_runs = _extract_data(exp)

    # 2. Normalize (Find Reference Point)
    min_val, max_val = normalize(ref, Fs)
    
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
                import warnings
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
        
    return MetricMatrix(mat, "Hypervolume", source_name=name)

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

def _calc_metric(exp, ref, MetricClass, name):
    if ref is None: ref = []
    if not isinstance(ref, list): ref = [ref]
    
    F_GENs, Fs, source_name, n_runs = _extract_data(exp)

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

def gd(exp, ref=None):
    if ref is None and hasattr(exp, 'optimal_front'):
        try:
            ref = exp.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"GD: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    return _calc_metric(exp, ref, GEN_gd, "GD")

def gdplus(exp, ref=None):
    if ref is None and hasattr(exp, 'optimal_front'):
        try:
            ref = exp.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"GD+: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    return _calc_metric(exp, ref, GEN_gdplus, "GD+")

def igd(exp, ref=None):
    if ref is None and hasattr(exp, 'optimal_front'):
        try:
            ref = exp.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"IGD: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    return _calc_metric(exp, ref, GEN_igd, "IGD")

def igdplus(exp, ref=None):
    if ref is None and hasattr(exp, 'optimal_front'):
        try:
            ref = exp.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"IGD+: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    return _calc_metric(exp, ref, GEN_igdplus, "IGD+")

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
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for mat in metric_matrices:
             data = mat.values
             label = f"{mat.metric_name} ({mat.source_name})" if mat.source_name else mat.metric_name
             
             if data.shape[1] > 1:
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)
                gens = np.arange(1, len(mean) + 1)
                v_min = np.nanmin(data, axis=1)
                v_max = np.nanmax(data, axis=1)
                
                ax.plot(gens, mean, label=label)
                ax.fill_between(gens, np.maximum(0, mean-std), mean+std, alpha=0.2)
                
                if show_bounds:
                    ax.plot(gens, v_min, '--', color=ax.get_lines()[-1].get_color(), alpha=0.5, linewidth=1)
                    ax.plot(gens, v_max, '--', color=ax.get_lines()[-1].get_color(), alpha=0.5, linewidth=1)
             else:
                ax.plot(np.arange(1, len(data)+1), data[:, 0], label=label)
        
        ax.set_title(final_title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(plot_name)
        ax.legend()
        plt.show()
        
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
        fig.show()

# TODO: Add IGD, GD, etc. same pattern.
