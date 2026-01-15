# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .scatter3d import Scatter3D
from .scatter2d import Scatter2D
import numpy as np

def spaceplot(*args, objectives=None, mode='auto', title=None, axis_labels=None):
    """
    Plots 3D scatter of objectives (Pareto Front).
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
            
    processed_args = []
    names = []
    trace_modes = [] # Store if we want markers or lines
    
    # Defaults
    if title is None: title = "Pareto-optimal front"
    if axis_labels is None: axis_labels = "Objective"
    
    from ..stats.attainment import AttainmentSurface

    for i, arg in enumerate(args):
        val = arg
        name = None
        t_mode = 'markers'
        
        # Unwrap standard MoeaBench objects
        # 1. AttainmentSurface (special case)
        if isinstance(arg, AttainmentSurface):
             val = arg
             name = arg.name
             t_mode = 'lines+markers' # Show the boundary clearly
        # 2. Prioritize front() method if available (Experiment/Run)
        elif hasattr(arg, 'front') and callable(getattr(arg, 'front')):
             val = arg.front()
        # 3. Fallback to .objectives property (Population)
        elif hasattr(arg, 'objectives'):
             val = arg.objectives
        
        # Try to extract name metadata
        if not name:
            # 1. inner array .name (SmartArray)
            if hasattr(val, 'name') and val.name:
                 name = val.name
            # 2. original object .name
            elif hasattr(arg, 'name') and arg.name:
                 name = arg.name
            # 3. .label (Population)
            elif hasattr(arg, 'label') and arg.label:
                 name = arg.label

        # Fallback name
        if not name:
             name = f"Data {i+1}"
             
        # Extract Plot Metadata from first argument
        if i == 0:
            if hasattr(val, 'label') and val.label and title == "Pareto-optimal front":
                 title = val.label
            if hasattr(val, 'axis_label') and val.axis_label and axis_labels == "Objective":
                 axis_labels = val.axis_label
        
        # Convert to numpy if needed
        val = np.array(val)
        processed_args.append(val)
        names.append(name)
        trace_modes.append(t_mode)
        
    # Axis selection
    if objectives is None:
        # Auto-detect data dimension
        dims = [d.shape[1] for d in processed_args if len(d.shape) > 1]
        max_dim = max(dims) if dims else 2
        
        if max_dim < 3:
             objectives = [0, 1]
        else:
             objectives = [0, 1, 2]
    
    # Selection of Plotter based on dimensions
    if len(objectives) == 2:
        s = Scatter2D(names, processed_args, objectives, type=title, mode=mode, axis_label=axis_labels, trace_modes=trace_modes)
    else:
        # Ensure 3rd dimension exists for Scatter3D
        for k in range(len(processed_args)):
             d = processed_args[k]
             if d.shape[1] < 3:
                 # Pad with zeros to ensure at least 3 columns for 3D plotting
                 padding = np.zeros((d.shape[0], 3 - d.shape[1]))
                 processed_args[k] = np.column_stack([d, padding])

        # Ensure objectives has 3 elements
        while len(objectives) < 3:
             objectives.append(0)
             
        s = Scatter3D(names, processed_args, objectives, type=title, mode=mode, axis_label=axis_labels, trace_modes=trace_modes)
    
    s.show()

def timeplot(*args, **kwargs):
    """
    Plots metric matrices over time.
    Wrapper for metrics.plot_matrix.
    """
    from ..metrics.evaluator import plot_matrix
    plot_matrix(args, **kwargs)

def rankplot(*args, labels=None, title=None, metric=None, height_scale=0.5, **kwargs):
    """
    Plots the 'Floating Rank' diagnostic with global normalization.
    
    Each dominance rank is represented by a vertical bar:
    - Vertical Position (Center): Measured Quality (e.g. Hypervolume).
    - Bar Height: Population Density (normalized count).
    
    Args:
        *args: StratificationResult objects.
        labels (list): Series labels.
        title (str): Plot title.
        metric: Callable metric function (default: mb.hypervolume).
        height_scale (float): Scaling factor for bar heights.
        **kwargs: Passed to the metric function.
    """
    import matplotlib.pyplot as plt
    from ..metrics.evaluator import hypervolume
    
    if metric is None:
        metric = hypervolume
        
    fig, ax = plt.subplots()
    
    # 0. Global Foundation: Collect all objectives to establish shared bounds
    all_objs_list = [res.objectives for res in args if hasattr(res, 'objectives') and res.objectives is not None]
    global_ref = np.vstack(all_objs_list) if all_objs_list else None
    
    # 1. Anchor Detection: Performance of the best experiment as 1.0 (Quality Scale)
    total_qualities = []
    for res in args:
        if hasattr(res, 'objectives') and res.objectives is not None:
            val = metric(res.objectives, ref=global_ref, **kwargs)
            total_qualities.append(float(val))
    anchor = max(total_qualities) if total_qualities else 1.0

    # 2. Peak Normalization: Find the most crowded rank across all series (Density Scale)
    # This ensures that the 'Search Effort' is visible regardless of population size.
    global_max_count = 0
    for res in args:
        if hasattr(res, 'ranks'):
            counts = res.frequencies()
            if len(counts) > 0:
                global_max_count = max(global_max_count, np.max(counts))
    if global_max_count == 0: global_max_count = 1

    n_series = len(args)
    width = 0.8 / n_series
    
    for i, res in enumerate(args):
        if not hasattr(res, 'ranks'): continue
        
        lbl = labels[i] if labels and i < len(labels) else getattr(res.source, 'name', f"Series {i+1}")
        color = plt.cm.tab10(i)
        
        # 3. Map: Calculate Quality per Rank using global foundation
        def _scalar_metric(objs, **m_kwargs):
            m_val = metric(objs, ref=global_ref, **m_kwargs)
            return float(m_val)
                
        # 4. Scale: Apply relative anchor
        q_raw = res.quality_by(_scalar_metric, **kwargs)
        q = q_raw / anchor
        
        # 5. Density: Apply Peak Normalization
        counts = res.frequencies()
        # Bar height is normalized relative to the global max count
        norm_heights = (counts / global_max_count) * height_scale
        
        ranks = np.arange(1, len(q) + 1)
        
        # 6. Plot: Floating Bars
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(ranks + offset, norm_heights, bottom=q - norm_heights/2, 
                      width=width, color=color, alpha=0.7, label=lbl, zorder=3)
        
        # 7. Labels: Add Floating Text for exact counts and quality
        # We try to determine the 'representative' population size to show real counts.
        n_runs = len(res.raw) if hasattr(res, 'raw') and res.raw is not None else 1
        pop_size = len(res.objectives) / n_runs if res.objectives is not None else 1
        
        for r_idx, (rank_val, quality_val, count_freq) in enumerate(zip(ranks, q, counts)):
            if np.isnan(quality_val): continue
            
            # Show avg count per run - very compact format
            real_n = int(round(count_freq * pop_size))
            txt = f"{real_n}\n{quality_val:.2f}"
            
            # Position text inside or next to the bar
            ax.text(rank_val + offset, quality_val, txt, 
                    ha='center', va='center', fontsize=7, color='black', 
                    weight='bold', zorder=5)

    max_r = max(res.max_rank for res in args if hasattr(res, 'max_rank'))
    ax.set_xticks(range(1, max_r + 1))
    ax.set_xlabel("Dominance Rank")
    ax.set_ylabel(f"Relative Strategy Efficiency ({metric.__name__ if hasattr(metric, '__name__') else 'Value'})")
    ax.set_title(title if title else "Floating Rank Profile (Relative Visibility)")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    
    # Dynamic Limit: Ensure bars + labels aren't clipped
    ax.set_ylim(bottom=0)
    _, top_curr = ax.get_ylim()
    ax.set_ylim(top=max(1.15, top_curr + 0.15))
    
    return ax
