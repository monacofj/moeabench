# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
from ..plotting.scatter3d import Scatter3D
from ..plotting.scatter2d import Scatter2D
from ..metrics.evaluator import plot_matrix, hypervolume
from ..stats.stratification import strata, StratificationResult, TierResult, tier
from ..stats.attainment import AttainmentSurface

def _resolve_to_result(args, target_type, resolve_fn):
    """
    Helper to resolve a mix of raw objects and results into results.
    """
    results = []
    labels = []
    for arg in args:
        if isinstance(arg, target_type):
            results.append(arg)
            labels.append(getattr(arg.source, 'name', 'Result'))
        else:
            # Assume it's a raw object (Experiment, Run, Pop)
            res = resolve_fn(arg)
            results.append(res)
            labels.append(getattr(arg, 'name', getattr(arg, 'label', 'Data')))
    return results, labels

def spaceplot(*args, objectives=None, mode='auto', title=None, axis_labels=None, labels=None):
    """
    [mb.view.spaceplot] Perspectiva Espacial.
    Plots 2D/3D scatter of objectives (Pareto Front).
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
    
    for i, arg in enumerate(args):
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
             t_mode = 'lines+markers'
        elif hasattr(arg, 'front') and callable(getattr(arg, 'front')):
             val = arg.front()
        elif hasattr(arg, 'objectives'):
             val = arg.objectives
        
        if not name:
            if hasattr(val, 'name') and val.name: name = val.name
            elif hasattr(arg, 'name') and arg.name: name = arg.name
            elif hasattr(arg, 'label') and arg.label: name = arg.label

        if not name: name = f"Data {i+1}"
             
        if i == 0:
            if hasattr(val, 'label') and val.label and title == "Pareto-optimal front":
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
    
    s.show()
    return s

def timeplot(*args, metric=None, **kwargs):
    """
    [mb.view.timeplot] Perspectiva Histórica.
    Plots metric matrices over time.
    """
    if metric is None:
        metric = hypervolume
        
    processed_args = []
    from ..metrics.evaluator import MetricMatrix
    
    for arg in args:
        if isinstance(arg, MetricMatrix):
            processed_args.append(arg)
        else:
            # Polymorphism: calculate metric if raw object passed
            processed_args.append(metric(arg))
            
    return plot_matrix(processed_args, **kwargs)

def rankplot(*args, title=None, **kwargs):
    """
    [mb.view.rankplot] Perspectiva Estrutural (Grounded).
    Plots frequency distribution across dominance ranks.
    """
    results, labels = _resolve_to_result(args, StratificationResult, strata)
    
    fig, ax = plt.subplots()
    n_series = len(results)
    width = 0.8 / n_series
    
    max_r = 0
    for i, res in enumerate(results):
        counts = res.frequencies()
        ranks = np.arange(1, len(counts) + 1)
        max_r = max(max_r, len(counts))
        
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(ranks + offset, counts, width=width, label=labels[i], alpha=0.7)
        
    ax.set_xticks(range(1, max_r + 1))
    ax.set_xlabel("Dominance Rank")
    ax.set_ylabel("Frequency (Population %)")
    ax.set_title(title if title else "Rank Structure (Selection Pressure)")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.show()
    return ax

def casteplot(*args, labels=None, title=None, metric=None, height_scale=0.5, **kwargs):
    """
    [mb.view.casteplot] Perspectiva Hierárquica (Floating).
    Plots Quality vs Density profile of dominance layers.
    """
    results, resolved_labels = _resolve_to_result(args, StratificationResult, strata)
    if labels is None: labels = resolved_labels
    
    if metric is None:
        metric = hypervolume
        
    fig, ax = plt.subplots()
    
    # Global Foundation: Collect all objectives to establish shared bounds
    all_objs_list = [res.objectives for res in results if hasattr(res, 'objectives') and res.objectives is not None]
    global_ref = np.vstack(all_objs_list) if all_objs_list else None
    
    # Anchor Detection
    total_qualities = []
    for res in results:
        if hasattr(res, 'objectives') and res.objectives is not None:
            val = metric(res.objectives, ref=global_ref, **kwargs)
            total_qualities.append(float(val))
    anchor = max(total_qualities) if total_qualities else 1.0

    # Peak Normalization
    global_max_count = 0
    for res in results:
        counts = res.frequencies()
        if len(counts) > 0:
            global_max_count = max(global_max_count, np.max(counts))
    if global_max_count == 0: global_max_count = 1

    n_series = len(results)
    width = 0.8 / n_series
    
    for i, res in enumerate(results):
        lbl = labels[i]
        color = plt.cm.tab10(i)
        
        def _scalar_metric(objs, **m_kwargs):
            m_val = metric(objs, ref=global_ref, **m_kwargs)
            return float(m_val)
                
        q_raw = res.quality_by(_scalar_metric, **kwargs)
        q = q_raw / anchor
        counts = res.frequencies()
        norm_heights = (counts / global_max_count) * height_scale
        ranks = np.arange(1, len(q) + 1)
        
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(ranks + offset, norm_heights, bottom=q - norm_heights/2, 
                      width=width, color=color, alpha=0.7, label=lbl, zorder=3)
        
        n_runs = len(res.raw) if hasattr(res, 'raw') and res.raw is not None else 1
        pop_size = len(res.objectives) / n_runs if res.objectives is not None else 1
        
        for r_idx, (rank_val, quality_val, count_freq) in enumerate(zip(ranks, q, counts)):
            if np.isnan(quality_val): continue
            real_n = int(round(count_freq * pop_size))
            txt = f"{real_n}\n{quality_val:.2f}"
            ax.text(rank_val + offset, quality_val, txt, 
                    ha='center', va='center', fontsize=7, color='black', 
                    weight='bold', zorder=5)

    max_r = max(res.max_rank for res in results)
    ax.set_xticks(range(1, max_r + 1))
    ax.set_xlabel("Dominance Rank (Caste)")
    ax.set_ylabel(f"Relative Strategy Efficiency ({metric.__name__ if hasattr(metric, '__name__') else 'Value'})")
    ax.set_title(title if title else "Caste Profile (Relative Performance Hierarchy)")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)
    _, top_curr = ax.get_ylim()
    ax.set_ylim(top=max(1.15, top_curr + 0.15))
    
    plt.show()
    return ax

def tierplot(exp1, exp2=None, title=None, **kwargs):
    """
    [mb.view.tierplot] Perspectiva Competitiva (Tier/Duel).
    Plots relative dominance proportion between two experiments (Stacked Bars).
    """
    if isinstance(exp1, TierResult):
        res = exp1
    else:
        res = tier(exp1, exp2)
        
    fig, ax = plt.subplots()
    nameA, nameB = res.group_labels
    
    ranks = np.arange(1, res.max_rank + 1)
    counts = np.array([res.tier_counts.get(r, 0) for r in ranks])
    propsA = np.array([res.joint_frequencies[r][0] for r in ranks])
    propsB = np.array([res.joint_frequencies[r][1] for r in ranks])
    
    valsA = counts * propsA
    valsB = counts * propsB
    
    ax.bar(ranks, valsA, label=nameA, color='C0', alpha=0.8)
    ax.bar(ranks, valsB, bottom=valsA, label=nameB, color='C1', alpha=0.8)
    
    ax.set_xticks(ranks)
    ax.set_xlabel("Global Tier Level")
    ax.set_ylabel("Population Count")
    ax.set_title(title if title else f"Competitive Tier Duel: {nameA} vs {nameB}")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add labels for proportions if clear
    for i, (pA, pB, vA, vB) in enumerate(zip(propsA, propsB, valsA, valsB)):
        if pA > 0.05:
            ax.text(i+1, vA/2, f"{pA*100:.0f}%", ha='center', va='center', fontsize=8, color='white', weight='bold')
        if pB > 0.05:
            ax.text(i+1, vA + vB/2, f"{pB*100:.0f}%", ha='center', va='center', fontsize=8, color='white', weight='bold')

    plt.show()
    return ax
