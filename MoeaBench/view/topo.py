# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
from ..defaults import defaults
from ..plotting.scatter3d import Scatter3D
from ..plotting.scatter2d import Scatter2D
from ..stats.topo_attainment import AttainmentSurface

def topo_shape(*args, objectives=None, mode='auto', title=None, axis_labels=None, labels=None, show=True, **kwargs):
    """
    [mb.view.topo_shape] Topographic Shape Perspective.
    Visualizes the geometry of the solution set (scatter/surface).
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
    if title is None: title = "Solution Set Geometry"
    if axis_labels is None: axis_labels = "Objective"
    
    new_args = list(args)

    for i, arg in enumerate(new_args):
        val = arg
        name = None
        t_mode = 'markers'
        
        # 1. Use explicit labels if provided
        if labels and i < len(labels):
            name = labels[i]

        # 2. Unwrap standard MoeaBench objects
        if isinstance(arg, AttainmentSurface):
             val = arg
             if not name: name = arg.name
             t_mode = 'lines'
        elif hasattr(arg, 'pf') and callable(getattr(arg, 'pf')) and not isinstance(arg, np.ndarray):
             # It's an experiment or problem, we want its front
             val = arg.front() if hasattr(arg, 'front') else arg.pf()
        elif hasattr(arg, 'front') and callable(getattr(arg, 'front')):
             val = arg.front()
        elif hasattr(arg, 'objectives'):
             val = arg.objectives
        
        if not name:
            # Smart Legend Naming
            # Try to get the name of the object (Experiment Name)
            obj_name = getattr(val, 'name', None) or getattr(arg, 'name', None)
            
            # Try to get the specific label (Filter)
            obj_label = getattr(val, 'label', None) or getattr(arg, 'label', None)
            
            if obj_name and obj_label and obj_name != obj_label:
                name = f"{obj_name} ({obj_label})"
            elif obj_name:
                name = obj_name
            elif obj_label:
                name = obj_label

        if not name: name = f"Data {i+1}"
             
        if i == 0:
            if hasattr(val, 'label') and val.label and title == "Solution Set Geometry":
                 title = val.label
            if hasattr(val, 'axis_label') and val.axis_label and axis_labels == "Objective":
                 axis_labels = val.axis_label
        
        val = np.array(val)
        processed_args.append(val)
        names.append(name)
        trace_modes.append(t_mode)
        
    if objectives is None:
        dims = [d.shape[1] for d in processed_args if len(d.shape) > 1]
        max_dim = max(dims) if dims else 2
        if max_dim < 3: objectives = [0, 1]
        else: objectives = [0, 1, 2]
    
    if len(objectives) == 2:
        s = Scatter2D(names, processed_args, objectives, type=title, mode=mode, axis_label=axis_labels, trace_modes=trace_modes)
    else:
        for k in range(len(processed_args)):
             d = processed_args[k]
             if d.shape[1] < 3:
                  padding = np.zeros((d.shape[0], 3 - d.shape[1]))
                  processed_args[k] = np.column_stack([d, padding])
        while len(objectives) < 3: objectives.append(0)
        s = Scatter3D(names, processed_args, objectives, type=title, mode=mode, axis_label=axis_labels, trace_modes=trace_modes)
    
    if show:
        s.show()
    return s

def topo_density(*args, axes=None, layout='grid', alpha=None, threshold=None, space='objs', title=None, show=True, ax=None, **kwargs):
    """
    [mb.view.topo_density] Spatial Distribution Perspective.
    Plots smooth Probability Density Estimates via Kernel Density Estimation (KDE)
    """
    from scipy.stats import gaussian_kde
    from ..stats.tests import topo_distribution as stats_topo_distribution, DistMatchResult
    
    # 1. Resolve Data and Labels
    buffers = []
    names = []
    
    for arg in args:
        if hasattr(arg, 'runs') and hasattr(arg, 'pop'):
            data = arg.front(**kwargs) if space == 'objs' else arg.set(**kwargs)
        else:
            data = arg
        buffers.append(np.asarray(data))
        names.append(getattr(arg, 'name', 'unnamed'))

    n_dims = buffers[0].shape[1]
    if axes is None:
        axes = list(range(min(n_dims, 4)))
    
    alpha = alpha if alpha is not None else defaults.alpha
    threshold = threshold if threshold is not None else defaults.displacement_threshold

    # 2. Statistical Analysis
    match_res = stats_topo_distribution(*args, space=space, axes=axes, method=kwargs.get('method', 'ks'), 
                                        alpha=alpha, threshold=threshold, **kwargs)
    
    # 3. Plotting Logic
    # 3. Plotting Logic
    if ax is not None:
        # Single axis injection mode
        if len(axes) != 1:
            raise ValueError("When 'ax' is provided, 'axes' must specify exactly one dimension.")
        axes_objs = [ax]
        layout = 'external'
    elif layout == 'grid':
        n_plots = len(axes)
        cols = 2 if n_plots > 1 else 1
        rows = (n_plots + 1) // 2
        fig, ax_grid = plt.subplots(rows, cols, figsize=(10, 4*rows), constrained_layout=True)
        axes_objs = np.atleast_1d(ax_grid).flatten()
        for j in range(len(axes), len(axes_objs)):
            axes_objs[j].axis('off')
    else:
        figures = []
        axes_objs = [None] * len(axes) 

    for i, ax_idx in enumerate(axes):
        if layout == 'independent':
            cur_fig, cur_ax = plt.subplots(figsize=(6, 4))
            figures.append(cur_fig)
        elif layout == 'external':
            cur_ax = axes_objs[i]
            cur_fig = cur_ax.figure
        else:
            cur_ax = axes_objs[i]
            cur_fig = fig
            
        axis_name = "f" if space == 'objs' else "x"
        cur_ax.set_xlabel(f"{axis_name}${ax_idx + 1}$")
        cur_ax.set_ylabel("Density (KDE)")
        
        ax_stats = match_res.results.get(ax_idx)
        p_val = getattr(ax_stats, 'p_value', 1.0) if ax_stats else 1.0
        d_stat = getattr(ax_stats, 'statistic', 0.0) if hasattr(ax_stats, 'statistic') else 0.0
        verdict = "Match" if p_val > alpha else "Mismatch"
        
        for b_idx, buffer in enumerate(buffers):
            sample = buffer[:, ax_idx]
            color = f"C{b_idx % 10}"
            sample = sample[np.isfinite(sample)]
            if len(np.unique(sample)) > 1:
                kde = gaussian_kde(sample)
                x_range = np.linspace(np.min(sample), np.max(sample), 100)
                y_vals = kde(x_range)
                lbl = f"{names[b_idx]}"
                if b_idx == 0: 
                     lbl += f" [{verdict}, p={p_val:.3f}]"
                cur_ax.plot(x_range, y_vals, color=color, label=lbl, lw=2)
                cur_ax.fill_between(x_range, y_vals, color=color, alpha=0.2)
            else:
                val = sample[0]
                cur_ax.axvline(val, color=color, label=names[b_idx], lw=2)
        
        cur_ax.set_title(f"Spatial Density: {axis_name}{ax_idx + 1}")
        cur_ax.legend(fontsize=8, frameon=True, facecolor='white', framealpha=0.9)
        cur_ax.grid(True, alpha=0.2, linestyle='--')
        
    if title: 
        if layout == 'external':
            pass # Title managed externally usually, or set specifically on ax
        else:
            fig.suptitle(title, fontsize=14)
            
    if show and layout != 'external': plt.show()
    return figures if layout == 'independent' else (ax if layout == 'external' else fig)

def topo_bands(*args, levels=[0.1, 0.5, 0.9], objectives=None, mode='auto', title=None, **kwargs):
    """
    [mb.view.topo_bands] Search Corridor Perspective.
    Visualizes reliability bands using Empirical Attainment Functions (EAF).
    """
    from ..stats.topo_attainment import topo_attainment
    
    surfaces = []
    for arg in args:
        if isinstance(arg, AttainmentSurface):
            surfaces.append(arg)
        else:
            # Calculate median surface by default if Experiment/Runs passed
            for lv in levels:
                surf = topo_attainment(arg, level=lv)
                surf.name = f"{getattr(arg, 'name', 'Exp')} ({lv*100:.0f}%)"
                surfaces.append(surf)
    
    if title is None: title = "Search Corridor (Attainment Bands)"
    
    return topo_shape(*surfaces, objectives=objectives, mode=mode, title=title, **kwargs)

def topo_gap(exp1, exp2, level=0.5, objectives=None, mode='auto', title=None, **kwargs):
    """
    [mb.view.topo_gap] Topologic Gap Perspective.
    Visualizes the spatial difference region between two algorithms.
    """
    from ..stats.topo_attainment import topo_gap as stats_topo_gap
    
    diff = stats_topo_gap(exp1, exp2, level=level)
    
    # To visualize the gap, we plot the two attainment surfaces that form it
    surf1 = diff.surf1
    surf2 = diff.surf2
    surf1.name = f"{getattr(exp1, 'name', 'Exp1')} ({level*100:.0f}%)"
    surf2.name = f"{getattr(exp2, 'name', 'Exp2')} ({level*100:.0f}%)"
    
    if title is None: 
        title = f"Topologic Gap ({level*100:.0f}% Attainment)"
        
    # TODO: In the future, specialized plotting for the AttainmentDiff object 
    # to highlight the area/volume between them.
    return topo_shape(surf1, surf2, objectives=objectives, mode=mode, title=title, **kwargs)
