# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
from ..stats.stratification import strata, StratificationResult, TierResult, tier
from ..metrics.evaluator import hypervolume

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

def strat_ranks(*args, title=None, **kwargs):
    """
    [mb.view.strat_ranks] Structural Perspective.
    Visualizes frequency distribution across dominance ranks.
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

def strat_caste(*args, labels=None, title=None, metric=None, height_scale=0.5, **kwargs):
    """
    [mb.view.strat_caste] Hierarchical Perspective.
    Visualizes Quality vs Density profile of dominance layers (classes).
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
        color = f"C{i % 10}"
        
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
    ax.set_xlabel("Dominance Rank (Class)")
    ax.set_ylabel(f"Relative Strategy Efficiency ({getattr(metric, '__name__', 'Value')})")
    ax.set_title(title if title else "Efficiency Hierarchy (Class Profile)")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)
    _, top_curr = ax.get_ylim()
    ax.set_ylim(top=max(1.15, top_curr + 0.15))
    
    plt.show()
    return ax

def strat_tiers(exp1, exp2=None, title=None, **kwargs):
    """
    [mb.view.strat_tiers] Competitive Perspective (Tier/Duel).
    Visualizes relative dominance proportion between two experiments.
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
