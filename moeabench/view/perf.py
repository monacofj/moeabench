# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..defaults import defaults
from ..metrics.evaluator import hypervolume
from ..core.base import emit_output
from ..core.display import show_matplotlib


def _perf_metric_error(method_name, metric, exc):
    metric_name = getattr(metric, "__name__", str(metric))
    lines = [
        f"{method_name}: incompatible metric or data for '{metric_name}'.",
        f"Reason: {type(exc).__name__}: {exc}",
        "Accepted metric profile:",
        "- callable metric(data, gens=..., **kwargs) returning MetricMatrix-like output",
        "Typical metrics:",
        "- mb.metrics.hv",
        "- mb.metrics.gd",
        "- mb.metrics.gdplus",
        "- mb.metrics.igd",
        "- mb.metrics.igdplus",
        "- mb.metrics.emd",
        "- mb.metrics.front_ratio",
    ]
    text = "\n".join(lines)
    md = "\n".join([
        f"**{method_name}**: incompatible metric or data for `{metric_name}`.",
        f"- **Reason**: `{type(exc).__name__}: {exc}`",
        "- **Accepted metric profile**: callable `metric(data, gens=..., **kwargs)` returning MetricMatrix-like output.",
        "- **Typical metrics**: `mb.metrics.hv`, `mb.metrics.gd`, `mb.metrics.gdplus`, `mb.metrics.igd`, `mb.metrics.igdplus`, `mb.metrics.emd`, `mb.metrics.front_ratio`.",
    ])
    emit_output(text, markdown=md)
    return None

def _infer_metric_axis_label(metric, args):
    """Infer the best metric label for axis titles from MetricMatrix inputs."""
    from ..metrics.evaluator import MetricMatrix
    for arg in args:
        if isinstance(arg, MetricMatrix):
            return getattr(arg, "metric_name", None) or getattr(metric, "__name__", "Value")
    return getattr(metric, "__name__", "Value")


def _plot_metric_matrices(metric_matrices, mode='auto', show_bounds=False, title=None, **kwargs):
    """Internal plotting engine for performance metric matrices."""
    if mode == 'auto':
        try:
            from IPython import get_ipython
            mode = 'interactive' if get_ipython() is not None else 'static'
        except (ImportError, NameError):
            mode = 'static'

    if not isinstance(metric_matrices, (list, tuple)):
        metric_matrices = [metric_matrices]
    if len(metric_matrices) == 1 and isinstance(metric_matrices[0], (list, tuple)):
        metric_matrices = metric_matrices[0]

    names = sorted(list(set(m.metric_name for m in metric_matrices)))
    plot_name = names[0] if len(names) == 1 else ", ".join(names)
    final_title = title if title else f"{plot_name} over Time"

    if mode == 'static':
        ax = kwargs.get('ax', None)
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        else:
            fig = ax.get_figure()

        lstyles = kwargs.get('linestyles', ['-'])
        if not isinstance(lstyles, (list, tuple)):
            lstyles = [lstyles]
        labels = kwargs.get('labels', [])

        for i, mat in enumerate(metric_matrices):
            data = mat.values
            if i < len(labels):
                label = labels[i]
            else:
                name = mat.source_name if mat.source_name else mat.metric_name
                G, R = data.shape
                meta = []
                if G == 1:
                    meta.append("G: 1")
                if R == 1:
                    meta.append("R: 1")
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
                ax.fill_between(gens, np.maximum(0, mean - std), mean + std, alpha=0.2)
                if show_bounds:
                    ax.plot(gens, v_min, '--', color=ax.get_lines()[-1].get_color(), alpha=0.5, linewidth=1)
                    ax.plot(gens, v_max, '--', color=ax.get_lines()[-1].get_color(), alpha=0.5, linewidth=1)
            else:
                ax.plot(np.arange(1, len(data) + 1), data[:, 0], label=label, linestyle=ls)

        ax.set_title(final_title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(plot_name)
        ax.legend()
        if kwargs.get('show', True) and kwargs.get('ax') is None:
            show_matplotlib(fig, auto_close=True)
        return fig, ax

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
            fig.add_trace(go.Scatter(x=gens, y=mean, mode='lines', name=label, line=dict(width=3)))
            lower_bound = np.maximum(0, mean - std)
            fig.add_trace(go.Scatter(
                x=np.concatenate([gens, gens[::-1]]),
                y=np.concatenate([mean + std, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(100, 100, 100, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))
            if show_bounds:
                fig.add_trace(go.Scatter(x=gens, y=v_min, mode='lines', line=dict(dash='dash', width=1), showlegend=False, opacity=0.5))
                fig.add_trace(go.Scatter(x=gens, y=v_max, mode='lines', line=dict(dash='dash', width=1), showlegend=False, opacity=0.5))
        else:
            fig.add_trace(go.Scatter(x=np.arange(1, len(data) + 1), y=data[:, 0], mode='lines', name=label))

    fig.update_layout(title=final_title, xaxis_title="Generation", yaxis_title=plot_name)
    if kwargs.get('show', True):
        fig.show()
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                return None
        except (ImportError, NameError):
            pass
    return fig

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

    try:
        for arg in args:
            if isinstance(arg, MetricMatrix):
                # If a matrix is already provided, slice it if gens is specified
                processed_args.append(arg[gens] if gens is not None else arg)
            else:
                # Polymorphism: calculate metric if raw object passed, passing gens
                processed_args.append(metric(arg, gens=gens, **metric_kwargs))
        return _plot_metric_matrices(processed_args, **kwargs)
    except Exception as exc:
        return _perf_metric_error("mb.view.perf_history", metric, exc)

def perf_spread(*args, metric=None, gen=-1, title=None, alpha=None, **kwargs):
    """
    [mb.view.perf_spread] Comparative Performance Perspective (Contrast).
    Visualizes comparative performance stats using Boxplots and 
    annotates Win Probability (A12) and Significance (p-value).
    """
    from ..stats import perf_shift, PerfCompareResult
    
    if len(args) < 1:
        emit_output(
            "mb.view.perf_spread requires at least one dataset.",
            markdown="**mb.view.perf_spread** requires at least one dataset."
        )
        return None

    alpha = alpha if alpha is not None else defaults.alpha

    if metric is None:
        metric = hypervolume
    metric_label = _infer_metric_axis_label(metric, args)

    stats_result = None
    if len(args) == 1 and isinstance(args[0], PerfCompareResult):
        stats_result = args[0]
        args = ()

    if stats_result is not None and getattr(stats_result, "samples", None):
        samples = [np.asarray(s).ravel() for s in stats_result.samples]
        labels = list(stats_result.labels or [f"Data {i+1}" for i in range(len(samples))])
        metric_label = stats_result.metric_label or metric_label
        if gen == -1 and getattr(stats_result, "gen", None) is not None:
            gen = stats_result.gen
    else:
        samples = []
        labels = []
    
    # Extract samples for plotting
    from ..metrics.evaluator import MetricMatrix
    metric_kwargs = kwargs.copy()
    if 'ref' not in metric_kwargs and len(args) > 1:
        # Keep cross-experiment comparability when raw experiments are provided.
        metric_kwargs['ref'] = list(args)
    
    # 1. Collect all distributions
    if not samples:
        try:
            for i, arg in enumerate(args):
                if isinstance(arg, MetricMatrix):
                    v = np.asarray(arg.gens(gen))
                elif isinstance(arg, (np.ndarray, list, tuple)):
                    v = np.asarray(arg)
                else:
                    res = metric(arg, **metric_kwargs)
                    if isinstance(res, MetricMatrix):
                        v = np.asarray(res.gens(gen))
                    else:
                        v = np.asarray(res)
                samples.append(v.ravel())
            
                # Legend Pattern: 'Name (run, gen)'
                exp_name = None
                run_idx = None
                if type(arg).__name__ == 'Run':
                    run_idx = getattr(arg, 'index', None)
                    if hasattr(arg, 'source'):
                        exp_name = getattr(arg.source, 'name', None)
                elif type(arg).__name__ == 'experiment':
                    exp_name = getattr(arg, 'name', None)
                elif isinstance(arg, MetricMatrix):
                    exp_name = getattr(arg, 'source_name', None) or getattr(arg, 'metric_name', None)
                    
                if not exp_name:
                    exp_name = getattr(arg, 'name', None) or f"Data {i+1}"
                    import re
                    exp_name = re.sub(r'\s*\(run\s*\d+\)', '', exp_name, flags=re.IGNORECASE)
                    
                pieces = []
                if run_idx is not None: pieces.append(f"run {run_idx}")
                if gen is not None and gen >= 0: pieces.append(f"gen {gen}")
                
                labels.append(f"{exp_name} ({', '.join(pieces)})" if pieces else exp_name)
        except Exception as exc:
            return _perf_metric_error("mb.view.perf_spread", metric, exc)
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(samples, labels=labels, patch_artist=True, 
                     medianprops={'color': 'black', 'linewidth': 2},
                     boxprops={'alpha': 0.7})
    
    # Colorize boxes
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(f"C{i % 10}")
        
    ax.set_ylabel(f"Performance ({metric_label})")
    ax.set_title(title if title else f"Performance Contrast (Gen {gen if gen >=0 else 'Final'})")
    ax.grid(True, axis='y', alpha=0.3)
    
    # 3. Annotate pairwise stats if exactly two
    if len(samples) == 2:
        if stats_result is None:
            if len(args) == 2:
                stats_result = perf_shift(args[0], args[1], metric=metric, gen=gen, **kwargs)

        prob = getattr(stats_result, "effect_size", None) if stats_result is not None else None
        p_val = getattr(stats_result, "p_value", None) if stats_result is not None else None
        sig_str = "*" if (p_val is not None and p_val < alpha) else "ns"
        
        # Position annotation between the two boxes
        y_max = max([np.max(s) for s in samples])
        y_range = y_max - min([np.min(s) for s in samples])
        ann_y = y_max + 0.05 * y_range
        
        ax.plot([1, 1, 2, 2], [ann_y, ann_y + 0.02 * y_range, ann_y + 0.02 * y_range, ann_y], 
                color='black', lw=1)
        if prob is not None:
            ann_text = f"A12={prob:.2f} ({sig_str})"
        elif p_val is not None:
            ann_text = f"p={p_val:.3f} ({sig_str})"
        else:
            ann_text = str(getattr(stats_result, "decision", "n/a"))
        ax.text(1.5, ann_y + 0.03 * y_range, ann_text, ha='center', va='bottom', fontsize=9, weight='bold')
        ax.set_ylim(top=ann_y + 0.15 * y_range)

    show_matplotlib(fig, auto_close=True)
    return ax

def perf_density(*args, metric=None, gen=-1, title=None, alpha=None, **kwargs):
    """
    [mb.view.perf_density] Performance Distribution Perspective.
    Visualizes metric probability density using KDE.
    """
    from scipy.stats import gaussian_kde
    from ..stats import perf_match, PerfCompareResult
    
    if len(args) < 1:
        emit_output(
            "mb.view.perf_density requires at least one dataset.",
            markdown="**mb.view.perf_density** requires at least one dataset."
        )
        return None

    alpha = alpha if alpha is not None else defaults.alpha

    if metric is None:
        metric = hypervolume
    metric_label = _infer_metric_axis_label(metric, args)

    stats_result = None
    if len(args) == 1 and isinstance(args[0], PerfCompareResult):
        stats_result = args[0]
        args = ()

    if stats_result is not None and getattr(stats_result, "samples", None):
        samples = [np.asarray(s).ravel() for s in stats_result.samples]
        names = list(stats_result.labels or [f"Data {i+1}" for i in range(len(samples))])
        metric_label = stats_result.metric_label or metric_label
        if gen == -1 and getattr(stats_result, "gen", None) is not None:
            gen = stats_result.gen
    else:
        samples = []
        names = []

    from ..metrics.evaluator import MetricMatrix
    metric_kwargs = kwargs.copy()
    if 'ref' not in metric_kwargs and len(args) > 1:
        # Keep cross-experiment comparability when raw experiments are provided.
        metric_kwargs['ref'] = list(args)
    if not samples:
        try:
            for i, arg in enumerate(args):
                if isinstance(arg, MetricMatrix):
                    v = np.asarray(arg.gens(gen))
                elif isinstance(arg, (np.ndarray, list, tuple)):
                    v = np.asarray(arg)
                else:
                    res = metric(arg, **metric_kwargs)
                    if isinstance(res, MetricMatrix):
                        v = np.asarray(res.gens(gen))
                    else:
                        v = np.asarray(res)
                samples.append(v.ravel())
            
                # Legend Pattern: 'Name (run, gen)'
                exp_name = None
                run_idx = None
                if type(arg).__name__ == 'Run':
                    run_idx = getattr(arg, 'index', None)
                    if hasattr(arg, 'source'):
                        exp_name = getattr(arg.source, 'name', None)
                elif type(arg).__name__ == 'experiment':
                    exp_name = getattr(arg, 'name', None)
                elif isinstance(arg, MetricMatrix):
                    exp_name = getattr(arg, 'source_name', None) or getattr(arg, 'metric_name', None)
                    
                if not exp_name:
                    exp_name = getattr(arg, 'name', None) or f"Data {i+1}"
                    import re
                    exp_name = re.sub(r'\s*\(run\s*\d+\)', '', exp_name, flags=re.IGNORECASE)
                    
                pieces = []
                if run_idx is not None: pieces.append(f"run {run_idx}")
                if gen is not None and gen >= 0: pieces.append(f"gen {gen}")
                
                names.append(f"{exp_name} ({', '.join(pieces)})" if pieces else exp_name)
        except Exception as exc:
            return _perf_metric_error("mb.view.perf_density", metric, exc)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # If exactly two, get p-value/verdict for legend.
    p_val = None
    verdict = None
    if len(samples) == 2:
        if stats_result is None:
            if len(args) == 2:
                stats_result = perf_match(args[0], args[1], metric=metric, gen=gen, **kwargs)

        p_val = getattr(stats_result, "p_value", None) if stats_result is not None else None
        decision = str(getattr(stats_result, "decision", "")).lower() if stats_result is not None else ""
        if decision == "match":
            verdict = "Match"
        elif decision in {"divergent", "mismatch"}:
            verdict = "Mismatch"
        elif p_val is not None:
            verdict = "Match" if p_val > alpha else "Mismatch"
    
    for i, sample in enumerate(samples):
        color = f"C{i %10}"
        sample = sample[np.isfinite(sample)]
        if len(np.unique(sample)) > 1:
            kde = gaussian_kde(sample)
            x_range = np.linspace(np.min(sample), np.max(sample), 100)
            y_vals = kde(x_range)
            
            lbl = f"{names[i]}"
            ax.plot(x_range, y_vals, color=color, label=lbl, lw=2)
            ax.fill_between(x_range, y_vals, color=color, alpha=0.2)
        else:
            ax.axvline(sample[0], color=color, label=names[i], lw=2)
            
    ax.set_xlabel(f"Performance ({metric_label})")
    ax.set_ylabel("Density (KDE)")
    ax.set_title(title if title else "Performance Distribution")
    if p_val is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle='None', color='none'))
        if verdict is not None:
            labels.append(f"(p={p_val:.3f}, {verdict})")
        else:
            labels.append(f"(p={p_val:.3f})")
        ax.legend(handles, labels, handlelength=0, handletextpad=0.0)
    else:
        ax.legend()
    ax.grid(True, alpha=0.2)
    
    show_matplotlib(fig, auto_close=True)
    return fig
