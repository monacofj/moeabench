# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
from ..stats.stratification import (
    strata,
    ranks as stats_ranks,
    caste as stats_caste,
    tiers as stats_tiers,
    StratificationResult,
    TierResult,
    CasteSummary,
    RankCompareResult,
    CasteCompareResult,
)
from ..metrics.evaluator import hypervolume
from ..core.display import show_matplotlib

def _resolve_to_result(args, target_type, resolve_fn):
    """
    Helper to resolve a mix of raw objects and results into results.
    """
    results = []
    labels = []
    for arg in args:
        if isinstance(arg, target_type):
            results.append(arg)
            name = getattr(arg, 'name', getattr(arg, 'label', None))
            if (not name or name == "Population") and hasattr(arg, 'source'):
                 parent = arg.source
                 name = getattr(parent, 'name', None)
            labels.append(name if name else "Result")
        else:
            # Assume it's a raw object (Experiment, Run, Pop, JoinedPop)
            res = resolve_fn(arg)
            results.append(res)
            # Metadata Inference logic
            name = getattr(arg, 'name', getattr(arg, 'label', None))
            if (not name or name == "Population") and hasattr(arg, 'source'):
                 # It's a Run/Pop/JoinedPop, try parent
                 parent = arg.source
                 name = getattr(parent, 'name', None)
                 if (not name or name == "experiment") and hasattr(parent, 'source'):
                     # Go up to Experiment
                     name = getattr(parent.source, 'name', None)
            
            # Final fallback to label if name is still missing or generic
            if not name or name == "experiment":
                name = getattr(arg, 'label', 'Data')
                
            labels.append(name)
    return results, labels

def strat_ranks(*args, title=None, show=True, **kwargs):
    """
    [mb.view.strat_ranks] Structural Perspective.
    Visualizes frequency distribution across dominance ranks.
    """
    if len(args) == 1 and isinstance(args[0], RankCompareResult):
        results = list(args[0].results)
        labels = list(args[0].labels)
    else:
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
    if show:
        show_matplotlib(fig)
    return ax

def strat_caste(*args, labels=None, title=None, metric=None, mode='collective', 
                 show_quartiles=True, show=True, **kwargs):
    """
    [mb.view.strat_caste] Advanced Hierarchical Perspective.
    Visualizes Quality vs Density profile using Variable-Width Boxplots.
    
    Args:
        mode (str): 'collective' (default) for aggregate rank quality (GDP),
                   'individual' for a distribution of individual solution merits (Per Capita).
        show_quartiles (bool): Whether to show small numeric labels for Q1, Q3 and Whiskers.
    """
    if metric is None:
        metric = hypervolume

    if len(args) == 1 and isinstance(args[0], CasteCompareResult):
        summaries = list(args[0].summaries)
        resolved_labels = list(args[0].labels)
        metric_name = args[0].metric_name
        mode = args[0].mode
    elif args and all(isinstance(arg, CasteSummary) for arg in args):
        summaries = list(args)
        resolved_labels = [s.name for s in summaries]
        metric_name = summaries[0].metric_name if summaries else getattr(metric, '__name__', 'Value')
    else:
        castes = stats_caste(*args, metric=metric, mode=mode, **kwargs)
        summaries = list(castes.summaries)
        resolved_labels = list(castes.labels)
        metric_name = castes.metric_name
        mode = castes.mode

    if labels is None:
        labels = resolved_labels

    fig, ax = plt.subplots(figsize=(8, 6))

    n_series = len(summaries)
    base_width = 0.7 / n_series
    
    # 2. Setup Plot Infrastructure (Rank Lanes)
    max_r = max((max(summary._data.keys()) if summary._data else 0) for summary in summaries)
    for r in range(1, max_r + 1):
        # Vertical Lane boundary
        if r < max_r:
            ax.axvline(r + 0.5, color='gray', linestyle='--', alpha=0.3, lw=0.8)
        # Background subtle shading for alternate lanes
        if r % 2 == 0:
            ax.axvspan(r - 0.5, r + 0.5, color='gray', alpha=0.05)

    for i, summary in enumerate(summaries):
        lbl = labels[i]
        color = f"C{i % 10}"
        # Group series tightly around the center of the rank (the integer)
        offset = (i - (n_series - 1) / 2) * (base_width * 1.1)

        rank_data = summary._data
        max_n = max((v['n'] for v in rank_data.values()), default=1.0)
        bxp_stats = []
        plot_widths = []
        plot_positions = []

        for r in range(1, max_r + 1):
            info = rank_data.get(r)
            if not info:
                continue
            w = (info['n'] / max_n) * base_width if max_n > 0 else base_width
            plot_widths.append(w)
            plot_positions.append(r + offset)
            q_min_whisker = info['min_w']
            q_25 = info['q1']
            q_50 = info['q']
            q_75 = info['q3']
            q_max_whisker = info['max_w']
            avg_n = info['n']
            bxp_stats.append({
                'med': q_50,
                'q1': q_25,
                'q3': q_75,
                'whislo': q_min_whisker,
                'whishi': q_max_whisker,
                'fliers': [],
            })
            
            # Helper for label positioning (Numbers only, no prefixes)
            def add_q_label(y_val, label_val, color, ha='left', va='center'):
                x_offset = w * 0.6 if ha == 'left' else -w * 0.6
                ax.text(r + offset + x_offset, y_val, f"{label_val:.2f}", 
                        ha=ha, va=va, fontsize=6, color=color, alpha=0.8)

            # Central primary label (average n and median q)
            ax.text(r + offset, q_50, f"{avg_n}\n{q_50:.2f}", 
                    ha='center', va='center', fontsize=7, color='black', 
                    weight='bold', zorder=10, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.2', lw=0.5))
            
            # Secondary specialized labels (Quartiles and Whiskers)
            if show_quartiles:
                # Push labels OUTWARD to avoid overlap between series
                side = 'right' if offset < 0 else 'left'
                add_q_label(q_max_whisker, q_max_whisker, color, ha=side)
                add_q_label(q_75, q_75, color, ha=side)
                add_q_label(q_25, q_25, color, ha=side)
                add_q_label(q_min_whisker, q_min_whisker, color, ha=side)

        if bxp_stats:
            bp = ax.bxp(
                bxp_stats,
                positions=plot_positions,
                widths=plot_widths,
                patch_artist=True,
                manage_ticks=False,
                medianprops={'color': 'black', 'linewidth': 1.5},
                boxprops={'facecolor': color, 'alpha': 0.6, 'edgecolor': color},
                whiskerprops={'color': color},
                capprops={'color': color},
            )

    ax.set_xticks(range(1, max_r + 1))
    ax.set_xlim(0.5, max_r + 0.5)
    ax.set_xlabel("Dominance Rank (Global Class)")
    ax.set_ylabel(f"Relative Quality Distribution ({metric_name})")
    ax.set_title(title if title else "Caste Distribution (Groups vs Tiers)")
    ax.legend([plt.Rectangle((0,0),1,1, color=f"C{i%10}", alpha=0.6) for i in range(n_series)], labels)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    if show:
        show_matplotlib(fig)
    return ax

def strat_tiers(exp1, exp2=None, title=None, show=True, **kwargs):
    """
    [mb.view.strat_tiers] Competitive Perspective (Tier/Duel).
    Visualizes relative dominance proportion between two experiments.
    """
    if isinstance(exp1, TierResult):
        res = exp1
    else:
        res = stats_tiers(exp1, exp2, **kwargs)
        
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

    if show:
        show_matplotlib(fig)
    return ax
