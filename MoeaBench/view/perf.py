# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
from ..defaults import defaults
from ..metrics.evaluator import plot_matrix, hypervolume

def perf_history(*args, metric=None, gens=None, **kwargs):
    """
    [mb.view.perf_history] Historic Performance Perspective.
    Visualizes metric evolution over time.
    """
    if metric is None:
        metric = hypervolume
        
    # Smart Arguments: Normalize gens (int -> slice)
    if gens is not None and isinstance(gens, int):
        if gens == -1:
            gens = slice(-1, None)
        else:
            gens = slice(gens)

    # Separate Metric Args from Plotting Args
    plot_keys = {'title', 'show_bounds', 'mode', 'show'} 
    metric_kwargs = {k: v for k, v in kwargs.items() if k not in plot_keys}

    # Dynamic Benchmarking (Context Injection): 
    # If multiple experiments are provided and no reference is specified, 
    # we use the union of all inputs as the benchmark to ensure comensurability.
    if 'ref' not in metric_kwargs and len(args) > 1:
        metric_kwargs['ref'] = list(args)

    processed_args = []
    from ..metrics.evaluator import MetricMatrix
    
    for arg in args:
        if isinstance(arg, MetricMatrix):
            # If a matrix is already provided, slice it if gens is specified
            processed_args.append(arg[gens] if gens is not None else arg)
        else:
            # Polymorphism: calculate metric if raw object passed, passing gens
            processed_args.append(metric(arg, gens=gens, **metric_kwargs))
            
    return plot_matrix(processed_args, **kwargs)

def perf_spread(*args, metric=None, gen=-1, title=None, alpha=None, **kwargs):
    """
    [mb.view.perf_spread] Comparative Performance Perspective (Contrast).
    Visualizes comparative performance stats using Boxplots and 
    annotates Win Probability (A12) and Significance (p-value).
    """
    from ..stats.tests import perf_evidence
    
    if len(args) < 1:
        raise ValueError("perf_spread requires at least one dataset.")

    alpha = alpha if alpha is not None else defaults.alpha

    if metric is None:
        metric = hypervolume
    
    # Extract samples for plotting
    samples = []
    labels = []
    
    # 1. Collect all distributions
    for i, arg in enumerate(args):
        # We reuse perf_evidence's _resolve_samples logic indirectly
        # but here we need them for plotting
        from ..stats.tests import _resolve_samples
        # Pairwise resolution vs others to get fair reference points if needed
        # For boxplots, we usually want to resolve all vs the global set
        v, _ = _resolve_samples(arg, args, metric=metric, gen=gen, **kwargs)
        samples.append(v)
        labels.append(getattr(arg, 'name', f"Data {i+1}"))
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(samples, labels=labels, patch_artist=True, 
                     medianprops={'color': 'black', 'linewidth': 2},
                     boxprops={'alpha': 0.7})
    
    # Colorize boxes
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(f"C{i % 10}")
        
    ax.set_ylabel(f"Performance ({getattr(metric, '__name__', 'Value')})")
    ax.set_title(title if title else f"Performance Contrast (Gen {gen if gen >=0 else 'Final'})")
    ax.grid(True, axis='y', alpha=0.3)
    
    # 3. Annotate pairwise stats if exactly two
    if len(args) == 2:
        res = perf_evidence(args[0], args[1], metric=metric, gen=gen, **kwargs)
        prob = res.perf_probability
        p_val = res.p_value
        sig_str = "*" if p_val < alpha else "ns"
        
        # Position annotation between the two boxes
        y_max = max([np.max(s) for s in samples])
        y_range = y_max - min([np.min(s) for s in samples])
        ann_y = y_max + 0.05 * y_range
        
        ax.plot([1, 1, 2, 2], [ann_y, ann_y + 0.02 * y_range, ann_y + 0.02 * y_range, ann_y], 
                color='black', lw=1)
        ax.text(1.5, ann_y + 0.03 * y_range, f"A12={prob:.2f} ({sig_str})", 
                ha='center', va='bottom', fontsize=9, weight='bold')
        ax.set_ylim(top=ann_y + 0.15 * y_range)

    plt.show()
    return ax

def perf_density(*args, metric=None, gen=-1, title=None, alpha=None, **kwargs):
    """
    [mb.view.perf_density] Performance Distribution Perspective.
    Visualizes metric probability density using KDE.
    """
    from scipy.stats import gaussian_kde
    from ..stats.tests import perf_distribution
    
    if len(args) < 1:
        raise ValueError("perf_density requires at least one dataset.")

    alpha = alpha if alpha is not None else defaults.alpha

    if metric is None:
        metric = hypervolume

    samples = []
    names = []
    from ..stats.tests import _resolve_samples
    for i, arg in enumerate(args):
        v, _ = _resolve_samples(arg, args, metric=metric, gen=gen, **kwargs)
        samples.append(v)
        names.append(getattr(arg, 'name', f"Data {i+1}"))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # If exactly two, get p-value for the legend
    p_val = None
    if len(args) == 2:
        res = perf_distribution(args[0], args[1], metric=metric, gen=gen, **kwargs)
        p_val = res.p_value
        verdict = "Match" if p_val > alpha else "Mismatch"
    
    for i, sample in enumerate(samples):
        color = f"C{i %10}"
        sample = sample[np.isfinite(sample)]
        if len(np.unique(sample)) > 1:
            kde = gaussian_kde(sample)
            x_range = np.linspace(np.min(sample), np.max(sample), 100)
            y_vals = kde(x_range)
            
            lbl = f"{names[i]}"
            if i == 0 and p_val is not None:
                lbl += f"\n(p={p_val:.3f})\n{verdict}"
            
            ax.plot(x_range, y_vals, color=color, label=lbl, lw=2)
            ax.fill_between(x_range, y_vals, color=color, alpha=0.2)
        else:
            ax.axvline(sample[0], color=color, label=names[i], lw=2)
            
    ax.set_xlabel(f"Performance ({getattr(metric, '__name__', 'Value')})")
    ax.set_ylabel("Density (KDE)")
    ax.set_title(title if title else "Performance Distribution")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    plt.show()
    return fig
def perf_front_size(*args, mode='run', title=None, **kwargs):
    """
    [mb.view.perf_front_size] Non-Dominated Density Perspective.
    Visualizes the evolution of the non-dominated front size (percentage) over generations.
    """
    from ..metrics.evaluator import front_size
    if title is None: 
        if str(mode).lower() == 'consensus':
            title = "Consensus Ratio (Superfront Density)"
        else:
            title = "Non-Dominated Population Ratio"
            
    # Wrap metric to pass mode
    metric_fn = lambda x, gens=None: front_size(x, mode=mode, gens=gens)
    metric_fn.__name__ = "Ratio"
    
    return perf_history(*args, metric=metric_fn, title=title, **kwargs)

def perf_hv(*args, title="Hypervolume Convergence", **kwargs):
    """
    [mb.view.perf_hv] Hypervolume Convergence Perspective.
    Visualizes Hypervolume evolution with automatic Dynamic Referencing.
    """
    from ..metrics.evaluator import hypervolume
    return perf_history(*args, metric=hypervolume, title=title, **kwargs)
