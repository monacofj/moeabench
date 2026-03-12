# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Optional, Any, Union
from ..defaults import defaults
from ..core.base import emit_output
from ..core.display import show_matplotlib
from ..diagnostics import audit, headway, closeness, coverage, gap, regularity, balance
from ..diagnostics import q_headway, q_closeness, q_coverage, q_gap, q_regularity, q_balance
from ..diagnostics import q_headway_points, q_closeness_points

def _resolve_mode(mode: str) -> str:
    """ Detects best mode if set to 'auto' based on environment and defaults. """
    if mode == 'auto':
        if defaults.backend == 'matplotlib': return 'static'
        if defaults.backend == 'plotly': return 'interactive'
        
        try:
            from IPython import get_ipython
            if get_ipython() is not None: return 'interactive'
        except (ImportError, NameError):
            pass
        return 'static'
    return mode

def _resolve_metric_data(target: Any, ground_truth: Optional[np.ndarray], metric_name: str, **kwargs):
    """ Internal helper to resolve physical facts and q-scores for a specific metric. """
    metric_name = metric_name.lower()
    
    # 1. Map metric name to functions
    f_map = {
        "closeness": closeness,
        "headway": headway,
        "coverage": coverage,
        "gap": gap,
        "regularity": regularity,
        "balance": balance
    }
    q_map = {
        "closeness": q_closeness,
        "headway": q_headway,
        "coverage": q_coverage,
        "gap": q_gap,
        "regularity": q_regularity,
        "balance": q_balance
    }
    
    if metric_name not in f_map:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(f_map.keys())}")
        
    f_func = f_map[metric_name]
    q_func = q_map[metric_name]
    
    # 2. Extract Problem Name (for Q-Score lookup)
    mop_name = "Unknown"
    mop_obj = None
    if hasattr(target, 'mop'): mop_obj = target.mop
    elif hasattr(target, 'problem'): mop_obj = target.problem
    
    if mop_obj:
        if hasattr(mop_obj, 'name'): mop_name = mop_obj.name
        else: mop_name = mop_obj.__class__.__name__
    elif hasattr(target, 'mop_name'):
        mop_name = target.mop_name
    elif isinstance(target, np.ndarray) and hasattr(target, 'name'):
        mop_name = target.name

    # 3. Compute Physical Fact
    f_val = f_func(target, ref=ground_truth, **kwargs)
    
    # 4. Compute Q-Score
    # Most Q-funcs take (f_val, problem, k)
    K = len(target.objectives) if hasattr(target, 'objectives') else 100 # Fallback
    q_res = q_func(f_val, problem=mop_name, k=K)
    
    return f_val, q_res

def clinic_ecdf(target: Any, ground_truth: Optional[np.ndarray] = None, metric: str = "closeness", mode: str = 'auto', show: bool = True, title: Optional[str] = None, **kwargs):
    """
    [mb.view.clinic_ecdf] Clinical CDF Plot.
    Focuses on goal-attainment and Headway (95th percentile).
    """
    mode = _resolve_mode(mode)
    f_res, q_res = _resolve_metric_data(target, ground_truth, metric, **kwargs)
    
    # Use raw_data from the FairResult if available, otherwise fallback to scalar array
    data = f_res.raw_data if hasattr(f_res, 'raw_data') and f_res.raw_data is not None else np.array([float(f_res)])
    
    sorted_data = np.sort(data)
    y = np.linspace(0, 1, len(sorted_data))
    m_val = np.median(data)
    h_val = np.percentile(data, 95)
    
    mop_name = q_res.problem if hasattr(q_res, 'problem') else "Unknown"
    auto_title = f"{metric.title()} Distribution: ECDF<br><sup>{q_res.description}</sup>"
    resolved_title = title if title is not None else auto_title
    x_label = f"Physical Fact (Layer 1) - [{metric}]"
    
    # Resolve plotting label (name)
    lbl = getattr(target, 'name', 'Experiment')
    
    if mode == 'static':
        fig = plt.figure(figsize=defaults.figsize)
        plt.step(sorted_data, y, where='post', label=lbl, color='teal', linewidth=1.5)
        
        # Drops for 50% (Median)
        m_val = np.median(data)
        plt.axhline(0.50, color='gray', linestyle=':', alpha=0.6, label='Median (50%)')
        plt.axvline(m_val, color='gray', linestyle=':', alpha=0.6)
        
        # Drops for 95% (Robust Max)
        plt.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95th Percentile')
        plt.axvline(h_val, color='red', linestyle='--', alpha=0.5)
        
        plt.title(resolved_title.replace('<br>', '\n').replace('<sup>', '').replace('</sup>', ''))
        plt.xlabel(x_label)
        plt.ylabel("Cumulative Probability")
        plt.grid(True, alpha=0.2)
        plt.legend(fontsize=9)
        if show: show_matplotlib(fig)
        return fig
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sorted_data, y=y, mode='lines', line=dict(shape='hv', color='teal', width=2), name=lbl))
        
        # Median drops
        m_val = np.median(data)
        fig.add_trace(go.Scatter(x=[0, m_val, m_val], y=[0.5, 0.5, 0], mode='lines', 
                                 line=dict(color='gray', dash='dot', width=1), name='Median (50%)'))
        
        # 95th Percentile drops
        fig.add_trace(go.Scatter(x=[0, h_val, h_val], y=[0.95, 0.95, 0], mode='lines', 
                                 line=dict(color='red', dash='dash', width=1), name='95th Percentile'))
        
        fig.update_layout(
            title=dict(text=resolved_title, x=0.5),
            xaxis_title=x_label, yaxis_title="Cumulative Probability",
            template=defaults.theme, width=defaults.plot_width, height=defaults.plot_height
        )
        if show: fig.show()
        return fig

def clinic_distribution(target: Any, ground_truth: Optional[np.ndarray] = None, metric: str = "closeness", mode: str = 'auto', show: bool = True, title: Optional[str] = None, **kwargs):
    """
    [mb.view.clinic_distribution] Morphological Error Plot.
    Focuses on the shape of the error (histogram/density).
    """
    mode = _resolve_mode(mode)
    f_res, q_res = _resolve_metric_data(target, ground_truth, metric, **kwargs)
    
    # Use raw_data from the FairResult if available, otherwise fallback to scalar array
    data = f_res.raw_data if hasattr(f_res, 'raw_data') and f_res.raw_data is not None else np.array([float(f_res)])

    # Statistics
    m_val = np.median(data)
    h_val = np.percentile(data, 95)

    auto_title = f"{metric.title()} Distribution: Point-wise Analysis<br><sup>{q_res.description}</sup>"
    resolved_title = title if title is not None else auto_title
    x_label = f"Physical Fact (Layer 1) - [{metric}]"

    if mode == 'static':
        fig = plt.figure(figsize=defaults.figsize)
        plt.hist(data, bins=32, density=True, alpha=0.6, color='skyblue', edgecolor='navy')
        plt.title(resolved_title.replace('<br>', '\n').replace('<sup>', '').replace('</sup>', ''))
        plt.xlabel(x_label)
        plt.ylabel("Density")
        plt.grid(True, alpha=0.2)
        if show: show_matplotlib(fig)
        return fig
    else:
        fig = go.Figure(data=[go.Histogram(x=data, histnorm='probability density', marker=dict(color='skyblue', line=dict(color='navy', width=1)), name=metric)])
        fig.update_layout(
            title=dict(text=resolved_title, x=0.5),
            xaxis_title=x_label, yaxis_title="Density",
            template=defaults.theme, width=defaults.plot_width, height=defaults.plot_height
        )
        if show: fig.show()
        return fig

def clinic_radar(target: Any, ground_truth: Optional[np.ndarray] = None, mode: str = 'auto', show: bool = True, title: Optional[str] = None, **kwargs):
    """
    [mb.view.clinic_radar] Clinical Fingerprint (Spider Plot).
    Visualizes the 6 Quality Scores (Q-Scores) in a single polygon.
    """
    mode = _resolve_mode(mode)

    # Polymorphic input:
    # 1) Pre-computed results (DiagnosticResult / QualityAuditResult / scores=)
    # 2) Raw targets (Experiment/Run/Population/ndarray), resolved via diagnostics.audit()
    scores = kwargs.get("scores", None)
    if scores is None:
        if hasattr(target, "q_audit_res") and getattr(target, "q_audit_res") is not None:
            scores = getattr(target.q_audit_res, "scores", None)
        elif hasattr(target, "scores"):
            scores = getattr(target, "scores", None)
        else:
            res = audit(target, ground_truth)
            if res and getattr(res, "q_audit_res", None):
                scores = getattr(res.q_audit_res, "scores", None)

    # Fallback for DiagnosticResult objects that carry only FAIR evidence.
    # We attempt to re-derive Q-scores from FAIR metrics when enough context is available.
    if not scores and hasattr(target, "fair_audit_res") and getattr(target, "fair_audit_res") is not None:
        fmetrics = getattr(target.fair_audit_res, "metrics", {}) or {}
        problem = kwargs.get("problem", "Unknown")
        k = int(kwargs.get("k", 100))
        try:
            if hasattr(target, "q_audit_res") and target.q_audit_res is not None:
                problem = getattr(target.q_audit_res, "mop_name", problem)
                k = int(getattr(target.q_audit_res, "k", k))
        except Exception:
            pass
        if hasattr(target, "diagnostic_context") and isinstance(target.diagnostic_context, dict):
            problem = target.diagnostic_context.get("problem", problem)
            k = int(target.diagnostic_context.get("k", k))

        if fmetrics:
            scores = {
                "Q_CLOSENESS": q_closeness(fmetrics.get("CLOSENESS"), problem=problem, k=k) if "CLOSENESS" in fmetrics else 0.0,
                "Q_COVERAGE": q_coverage(fmetrics.get("COVERAGE"), problem=problem, k=k) if "COVERAGE" in fmetrics else 0.0,
                "Q_GAP": q_gap(fmetrics.get("GAP"), problem=problem, k=k) if "GAP" in fmetrics else 0.0,
                "Q_REGULARITY": q_regularity(fmetrics.get("REGULARITY"), problem=problem, k=k) if "REGULARITY" in fmetrics else 0.0,
                "Q_BALANCE": q_balance(fmetrics.get("BALANCE"), problem=problem, k=k) if "BALANCE" in fmetrics else 0.0,
                "Q_HEADWAY": q_headway(fmetrics.get("HEADWAY"), problem=problem, k=k) if "HEADWAY" in fmetrics else 0.0,
            }

    if not scores:
        emit_output(
            "Warning: [mb.view.clinic_radar] no quality scores available for this input.",
            markdown="> Warning: `clinic_radar` could not resolve quality scores for this input."
        )
        return None

    cat = ["Q_CLOSENESS", "Q_COVERAGE", "Q_GAP", "Q_REGULARITY", "Q_BALANCE", "Q_HEADWAY"]
    labels = [c.replace("Q_", "").title() for c in cat]
    values = []
    for c in cat:
        if c not in scores:
            values.append(0.0)
            continue
        qv = scores[c]
        values.append(float(qv.value) if hasattr(qv, "value") else float(qv))

    resolved_title = title if title is not None else "Clinical Quality Fingerprint (Q-Scores)"
    
    if mode == 'static':
        angles = np.linspace(0, 2 * np.pi, len(cat), endpoint=False).tolist()
        v_closed = values + [values[0]]
        a_closed = angles + [angles[0]]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(a_closed, v_closed, color='teal', alpha=0.25)
        ax.plot(a_closed, v_closed, color='teal', linewidth=2)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        
        # Improve grid visibility
        ax.grid(True, alpha=0.5, color='gray', linestyle='--')
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8, color='gray')
        
        plt.title(resolved_title)
        if show: show_matplotlib(fig)
        return fig
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself', line=dict(color='teal'), name='Q-Scores'))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0], gridcolor='#E0E0E0'),
                angularaxis=dict(gridcolor='#E0E0E0')
            ),
            title=dict(text=resolved_title, x=0.5),
            template=defaults.theme, width=defaults.plot_width, height=defaults.plot_height
        )
        if show: fig.show()
        return fig

def clinic_history(target: Any, ground_truth: Optional[np.ndarray] = None, metric: str = "closeness", mode: str = 'auto', show: bool = True, gens: Optional[Union[int, slice]] = None, title: Optional[str] = None, **kwargs):
    """
    [mb.view.clinic_history] Temporal Health Chart.
    Visualizes the evolution of a metric over generations.
    """
    # Smart Arguments: Normalize gens (int -> slice)
    if gens is not None and isinstance(gens, int):
        if gens == -1:
            gens = slice(-1, None)
        else:
            gens = slice(gens)

    mode = _resolve_mode(mode)
    is_exp = hasattr(target, 'runs')
    is_run = hasattr(target, 'history')
    if not (is_exp or is_run) or not hasattr(target, 'mop'):
        emit_output(
            "Warning: clinic_history requires an Experiment or Run with history enabled.",
            markdown="> Warning: `clinic_history` requires an `Experiment` or `Run` with history enabled."
        )
        return None
        
    mop = target.mop
    GT = ground_truth if ground_truth is not None else mop.pf()
    f_func = {"closeness": closeness, "headway": headway, "coverage": coverage, "gap": gap, "regularity": regularity, "balance": balance}.get(metric.lower())
    resolved_title = title if title is not None else f"Clinic History: {metric.upper()} Evolution"
    
    runs = target if hasattr(target, '__iter__') else [target]
    
    if mode == 'static':
        fig = plt.figure(figsize=defaults.figsize)
        for run in runs:
            # We convert the FairResult to float, which extracts the representative .value
            hist = run.history('f')
            if gens is not None:
                hist = hist[gens]
                if isinstance(hist, np.ndarray) and hist.ndim == 1: hist = [hist]

            timeseries = [float(f_func(pop_f, ref=GT, **kwargs)) for pop_f in hist]
            plt.plot(range(len(timeseries)), timeseries, label=run.name)
        plt.title(resolved_title)
        plt.xlabel("Generation"); plt.ylabel(f"Physical Fact [{metric}]")
        plt.grid(True, alpha=0.3); plt.legend()
        if show: show_matplotlib(fig)
        return fig
    else:
        fig = go.Figure()
        for run in runs:
             # We convert the FairResult to float, which extracts the representative .value
            hist = run.history('f')
            if gens is not None:
                hist = hist[gens]
                if isinstance(hist, np.ndarray) and hist.ndim == 1: hist = [hist]

            timeseries = [float(f_func(pop_f, ref=GT, **kwargs)) for pop_f in hist]
            fig.add_trace(go.Scatter(x=list(range(len(timeseries))), y=timeseries, mode='lines', name=run.name))
        fig.update_layout(
            title=dict(text=resolved_title, x=0.5),
            xaxis_title="Generation", yaxis_title=f"Physical Fact [{metric}]",
            template=defaults.theme, width=defaults.plot_width, height=defaults.plot_height
        )
        if show: fig.show()
        return fig
