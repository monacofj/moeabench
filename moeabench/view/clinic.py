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
from .style import MOEABENCH_PALETTE, GRID_COLOR, MEDIAN_COLOR, ALERT_COLOR, TEXT_MUTED
from ..diagnostics import audit, headway, closeness, coverage, gap, regularity, balance
from ..diagnostics import q_headway, q_closeness, q_coverage, q_gap, q_regularity, q_balance
from ..diagnostics import q_headway_points, q_closeness_points
from ..diagnostics.base import DiagnosticValue

RADAR_GRID_COLOR = "#B8C2CF"
RADAR_RING_LABEL = "#5B6574"
DIST_GRID_COLOR = "#C7D0DB"
DIST_BAR_EDGE = "#EEF2F7"

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
    """Internal helper to resolve the physical diagnostic result for one metric."""
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
    if metric_name not in f_map:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(f_map.keys())}")
        
    f_func = f_map[metric_name]
    if isinstance(target, DiagnosticValue):
        return target
    return f_func(target, ref=ground_truth, **kwargs)


def _resolve_radar_scores(target: Any, ground_truth: Optional[np.ndarray] = None, **kwargs):
    """Resolve radar-ready Q-scores plus a display label for one input."""
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

    label = (
        getattr(target, "experiment_name", None)
        or getattr(target, "name", None)
        or getattr(getattr(target, "source", None), "name", None)
        or "Q-Scores"
    )
    return scores, label

def clinic_ecdf(target: Any, ground_truth: Optional[np.ndarray] = None, metric: str = "closeness", mode: str = 'auto', show: bool = True, title: Optional[str] = None, **kwargs):
    """
    [mb.view.clinic_ecdf] Clinical CDF Plot.
    Focuses on goal-attainment and Headway (95th percentile).
    """
    mode = _resolve_mode(mode)
    f_res = _resolve_metric_data(target, ground_truth, metric, **kwargs)
    data = f_res.samples
    sorted_data = f_res.sorted_samples
    y = f_res.ecdf_y
    m_val = f_res.median
    h_val = f_res.p95

    auto_title = f"{metric.title()} Distribution: ECDF<br><sup>{f_res.description}</sup>"
    resolved_title = title if title is not None else auto_title
    x_label = f"Physical Fact (Layer 1) - [{metric}]"
    
    # Resolve plotting label (name)
    lbl = getattr(target, 'name', None) or getattr(getattr(target, 'source', None), 'name', None) or 'Experiment'
    
    if mode == 'static':
        fig = plt.figure(figsize=defaults.figsize)
        plt.step(sorted_data, y, where='post', label=lbl, color=MOEABENCH_PALETTE[0], linewidth=1.5)
        
        # Drops for 50% (Median)
        m_val = np.median(data)
        plt.axhline(0.50, color=MEDIAN_COLOR, linestyle=':', alpha=0.7, label='Median (50%)')
        plt.axvline(m_val, color=MEDIAN_COLOR, linestyle=':', alpha=0.7)
        
        # Drops for 95% (Robust Max)
        plt.axhline(0.95, color=ALERT_COLOR, linestyle='--', alpha=0.65, label='95th Percentile')
        plt.axvline(h_val, color=ALERT_COLOR, linestyle='--', alpha=0.65)
        
        plt.title(resolved_title.replace('<br>', '\n').replace('<sup>', '').replace('</sup>', ''))
        plt.xlabel(x_label)
        plt.ylabel("Cumulative Probability")
        plt.grid(True, alpha=0.2, color=GRID_COLOR)
        plt.legend(fontsize=9)
        if show: show_matplotlib(fig, auto_close=True)
        return fig
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted_data, y=y, mode='lines',
            line=dict(shape='hv', color=MOEABENCH_PALETTE[0], width=2), name=lbl
        ))
        
        # Median drops
        m_val = np.median(data)
        fig.add_trace(go.Scatter(x=[0, m_val, m_val], y=[0.5, 0.5, 0], mode='lines', 
                                 line=dict(color=MEDIAN_COLOR, dash='dot', width=1), name='Median (50%)'))
        
        # 95th Percentile drops
        fig.add_trace(go.Scatter(x=[0, h_val, h_val], y=[0.95, 0.95, 0], mode='lines', 
                                 line=dict(color=ALERT_COLOR, dash='dash', width=1), name='95th Percentile'))
        
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
    f_res = _resolve_metric_data(target, ground_truth, metric, **kwargs)
    data = f_res.samples
    auto_title = f"{metric.title()} Distribution: Point-wise Analysis<br><sup>{f_res.description}</sup>"
    resolved_title = title if title is not None else auto_title
    x_label = f"Physical Fact (Layer 1) - [{metric}]"

    if mode == 'static':
        fig = plt.figure(figsize=defaults.figsize)
        plt.hist(
            data,
            bins=32,
            density=True,
            alpha=0.88,
            color=MOEABENCH_PALETTE[0],
            edgecolor=DIST_BAR_EDGE,
            linewidth=0.9,
            zorder=3
        )
        plt.title(resolved_title.replace('<br>', '\n').replace('<sup>', '').replace('</sup>', ''))
        plt.xlabel(x_label)
        plt.ylabel("Density")
        ax = plt.gca()
        ax.set_axisbelow(True)
        plt.grid(True, axis='y', alpha=0.45, color=DIST_GRID_COLOR, linestyle='--', linewidth=0.8)
        plt.grid(True, axis='x', alpha=0.18, color=DIST_GRID_COLOR, linestyle=':', linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if show: show_matplotlib(fig, auto_close=True)
        return fig
    else:
        fig = go.Figure(data=[go.Histogram(
            x=data,
            histnorm='probability density',
            marker=dict(color=MOEABENCH_PALETTE[0], line=dict(color="rgba(238,242,247,0.95)", width=1)),
            opacity=0.86,
            name=metric
        )])
        fig.update_layout(
            title=dict(text=resolved_title, x=0.5),
            xaxis_title=x_label, yaxis_title="Density",
            template=defaults.theme,
            width=defaults.plot_width,
            height=defaults.plot_height,
            xaxis=dict(showgrid=True, gridcolor='rgba(199,208,219,0.22)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(199,208,219,0.55)')
        )
        if show: fig.show()
        return fig

def clinic_radar(*targets: Any, ground_truth: Optional[np.ndarray] = None, mode: str = 'auto', show: bool = True, title: Optional[str] = None, **kwargs):
    """
    [mb.view.clinic_radar] Clinical Fingerprint (Spider Plot).
    Visualizes one or more sets of the 6 Quality Scores in a radar chart.
    """
    mode = _resolve_mode(mode)
    if not targets:
        return None

    cat = ["Q_CLOSENESS", "Q_COVERAGE", "Q_GAP", "Q_REGULARITY", "Q_BALANCE", "Q_HEADWAY"]
    labels = [c.replace("Q_", "").title() for c in cat]
    series = []
    for target in targets:
        scores, label = _resolve_radar_scores(target, ground_truth, **kwargs)
        if not scores:
            emit_output(
                "Warning: [mb.view.clinic_radar] no quality scores available for one input.",
                markdown="> Warning: `clinic_radar` could not resolve quality scores for one input."
            )
            continue

        values = []
        for c in cat:
            if c not in scores:
                values.append(0.0)
                continue
            qv = scores[c]
            values.append(float(qv.value) if hasattr(qv, "value") else float(qv))
        series.append((label, values))

    if not series:
        return None

    if title is not None:
        resolved_title = title
    elif len(series) == 1:
        resolved_title = f"Clinical Quality Fingerprint ({series[0][0]})"
    else:
        names = " vs ".join(label for label, _ in series)
        resolved_title = f"Clinical Quality Fingerprints ({names})"
    
    if mode == 'static':
        angles = np.linspace(0, 2 * np.pi, len(cat), endpoint=False).tolist()
        a_closed = angles + [angles[0]]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', MOEABENCH_PALETTE)
        for idx, (label, values) in enumerate(series):
            color = palette[idx % len(palette)]
            v_closed = values + [values[0]]
            ax.fill(a_closed, v_closed, color=color, alpha=0.25 if len(series) == 1 else 0.15)
            ax.plot(a_closed, v_closed, color=color, linewidth=2, label=label)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, color=RADAR_RING_LABEL)
        ax.set_ylim(0, 1.0)
        
        # Improve grid visibility
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.75, color=RADAR_GRID_COLOR, linestyle='--', linewidth=0.9)
        ax.spines['polar'].set_color(RADAR_GRID_COLOR)
        ax.spines['polar'].set_linewidth(1.0)
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8, color=RADAR_RING_LABEL)
        
        plt.title(resolved_title)
        if len(series) > 1:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
        if show: show_matplotlib(fig, auto_close=True)
        return fig
    else:
        fig = go.Figure()
        palette = MOEABENCH_PALETTE
        for idx, (label, values) in enumerate(series):
            color = palette[idx % len(palette)]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                fillcolor=color,
                opacity=0.18 if len(series) > 1 else 0.25,
                line=dict(color=color),
                name=label
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.25, 0.5, 0.75, 1.0],
                    gridcolor='rgba(184,194,207,0.9)',
                    linecolor='rgba(184,194,207,0.95)',
                    tickfont=dict(color=RADAR_RING_LABEL)
                ),
                angularaxis=dict(
                    gridcolor='rgba(184,194,207,0.8)',
                    linecolor='rgba(184,194,207,0.95)',
                    tickfont=dict(color=RADAR_RING_LABEL)
                )
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
    if isinstance(target, DiagnosticValue) and target.history_values is not None:
        values = np.asarray(target.history_values, dtype=float)
        labels = list(target.history_labels or ["Series"])
        if gens is not None:
            values = values[gens]
            if values.ndim == 1:
                values = values.reshape(-1, 1)
        resolved_title = title if title is not None else f"Clinic History: {target.name} Evolution"

        if mode == 'static':
            fig = plt.figure(figsize=defaults.figsize)
            for j in range(values.shape[1]):
                plt.plot(range(values.shape[0]), values[:, j], label=labels[j])
            plt.title(resolved_title)
            plt.xlabel("Generation"); plt.ylabel(f"Physical Fact [{target.name.lower()}]")
            plt.grid(True, alpha=0.3); plt.legend()
            if show: show_matplotlib(fig, auto_close=True)
            return fig
        else:
            fig = go.Figure()
            for j in range(values.shape[1]):
                fig.add_trace(go.Scatter(x=list(range(values.shape[0])), y=values[:, j], mode='lines', name=labels[j]))
            fig.update_layout(
                title=dict(text=resolved_title, x=0.5),
                xaxis_title="Generation", yaxis_title=f"Physical Fact [{target.name.lower()}]",
                template=defaults.theme, width=defaults.plot_width, height=defaults.plot_height
            )
            if show: fig.show()
            return fig

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
        if show: show_matplotlib(fig, auto_close=True)
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
