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

def strat_caste(*args, labels=None, title=None, metric=None, mode='collective', 
                 show_quartiles=True, **kwargs):
    """
    [mb.view.strat_caste] Advanced Hierarchical Perspective.
    Visualizes Quality vs Density profile using Variable-Width Boxplots.
    
    Args:
        mode (str): 'collective' (default) for aggregate rank quality (GDP),
                   'individual' for a distribution of individual solution merits (Per Capita).
        show_quartiles (bool): Whether to show small numeric labels for Q1, Q3 and Whiskers.
    """
    # Allow legacy alias 'rank_caste2' to point here too if needed, but the canonical name is now strat_caste
    results, resolved_labels = _resolve_to_result(args, StratificationResult, strata)
    if labels is None: labels = resolved_labels
    
    if metric is None:
        metric = hypervolume
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Global Foundation
    all_objs_list = [res.objectives for res in results if hasattr(res, 'objectives') and res.objectives is not None]
    global_ref = np.vstack(all_objs_list) if all_objs_list else None
    
    total_qualities = []
    for res in results:
        if hasattr(res, 'objectives') and res.objectives is not None:
            val = metric(res.objectives, ref=global_ref, **kwargs)
            total_qualities.append(float(val))
    anchor = max(total_qualities) if total_qualities else 1.0

    n_series = len(results)
    base_width = 0.7 / n_series
    
    # 2. Setup Plot Infrastructure (Rank Lanes)
    max_r = max(res.max_rank for res in results)
    for r in range(1, max_r + 1):
        # Vertical Lane boundary
        if r < max_r:
            ax.axvline(r + 0.5, color='gray', linestyle='--', alpha=0.3, lw=0.8)
        # Background subtle shading for alternate lanes
        if r % 2 == 0:
            ax.axvspan(r - 0.5, r + 0.5, color='gray', alpha=0.05)

    for i, res in enumerate(results):
        lbl = labels[i]
        color = f"C{i % 10}"
        
        def _scalar_metric(objs, **m_kwargs):
            m_val = metric(objs, ref=global_ref, **m_kwargs)
            return float(m_val)
                
        counts = res.frequencies()
        ranks = np.arange(1, len(counts) + 1)
        # Group series tightly around the center of the rank (the integer)
        offset = (i - (n_series - 1) / 2) * (base_width * 1.1)
        
        plot_data = []
        plot_widths = []
        plot_positions = []
        
        max_freq = np.max(counts) if len(counts) > 0 else 1.0
        
        for r in range(1, len(counts) + 1):
            mask = (res.rank_array == r)
            if not np.any(mask): 
                continue
                
            sub_objs = res.objectives[mask]
            
            if mode == 'collective':
                # GDP Mode: Distribution of total rank quality across runs
                if hasattr(res, 'sub_results') and res.sub_results is not None:
                    # Multi-run data: use pre-calculated sub-results
                    samples = []
                    for sub_res in res.sub_results:
                        sub_mask = (sub_res.rank_array == r)
                        if np.any(sub_mask):
                             samples.append(_scalar_metric(sub_res.objectives[sub_mask]) / anchor)
                        else:
                             samples.append(0.0)
                else:
                    # Single run: Collective is just a single point (the total rank quality)
                    samples = [_scalar_metric(sub_objs) / anchor]
            else:
                # Individual Mode (Per Capita): Distribution of individual merit
                samples = [_scalar_metric(sub_objs[j:j+1]) / anchor for j in range(len(sub_objs))]
            
            plot_data.append(samples)
            # Width proportional to frequency, but constrained to its lane share
            w = (counts[r-1] / max_freq) * base_width
            plot_widths.append(w)
            plot_positions.append(r + offset)
            
            # Numeric Annotation (Statistical Summary)
            q_stats = np.percentile(samples, [25, 50, 75])
            q_25, q_50, q_75 = q_stats
            iqr = q_75 - q_25
            
            # Standard Whisker Logic (1.5 * IQR)
            upper_whisker_limit = q_75 + 1.5 * iqr
            lower_whisker_limit = q_25 - 1.5 * iqr
            
            # Actual whisker points are the most extreme values within the limits
            q_max_whisker = np.max([s for s in samples if s <= upper_whisker_limit]) if any(samples <= upper_whisker_limit) else q_75
            q_min_whisker = np.min([s for s in samples if s >= lower_whisker_limit]) if any(samples >= lower_whisker_limit) else q_25
            
            # Calculate average n per run for this rank
            if hasattr(res, 'sub_results') and res.sub_results:
                # Count actual solutions in this rank for each run
                avg_n = int(round(np.mean([np.sum(s.rank_array == r) for s in res.sub_results])))
            else:
                avg_n = int(round(counts[r-1] * len(res.objectives)))
            
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

        if plot_data:
            bp = ax.boxplot(plot_data, positions=plot_positions, widths=plot_widths, 
                            patch_artist=True, manage_ticks=False,
                            medianprops={'color': 'black', 'linewidth': 1.5},
                            boxprops={'facecolor': color, 'alpha': 0.6, 'edgecolor': color})

    ax.set_xticks(range(1, max_r + 1))
    ax.set_xlim(0.5, max_r + 0.5)
    ax.set_xlabel("Dominance Rank (Global Class)")
    ax.set_ylabel(f"Relative Quality Distribution ({getattr(metric, '__name__', 'Value')})")
    ax.set_title(title if title else "Caste Distribution (Groups vs Tiers)")
    ax.legend([plt.Rectangle((0,0),1,1, color=f"C{i%10}", alpha=0.6) for i in range(n_series)], labels)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.2)
    
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
