#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 15: Individual vs Grid-Aggregated Hypervolume Perspectives
-------------------------------------------------------------
This example demonstrates the three scaling modes in MoeaBench, controlled by 
the `scale` parameter:

1. 'raw': Physical volume dominated in the objective space ($H_{raw}$). 
   Measures absolute search progress in objective units.
2. 'relative': Aggregated Efficiency ($H_{rel}$). Scaled [0, 1] based on 
   the range of all solutions present in the session (or specified in `ref`).
3. 'absolute': Theoretical Optimality ($H_{abs}$). Scaled [0, 1] relative 
   to the theoretical maximum defined by the Ground Truth.

We also explore the `joint` parameter, which controls whether algorithms 
share the same Bounding Box (BBox) during auto-normalization.
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt
import numpy as np

def main():
    print(f"MoeaBench v{mb.system.version()}")
    print("Example 15: Individual vs Grid-Aggregated Hypervolume Perspectives")
    print("========================================================\n")

    # 1. Setup the Problem & Calibration
    dtlz2 = mb.mops.DTLZ2(n_var=7, n_obj=3)
    dtlz2.calibrate()
    
    # Configuration A: Baseline (Low budget)
    print("\nRunning Experiment 1 (Baseline: 20 individuals)...")
    exp1 = mb.experiment(dtlz2, mb.moeas.NSGA2(population=20, generations=50))
    exp1.name = "Baseline NSGA-II"
    exp1.run(repeat=5)
    
    # Configuration B: Premium (High budget)
    print("\nRunning Experiment 2 (Premium: 100 individuals)...")
    exp2 = mb.experiment(dtlz2, mb.moeas.NSGA2(population=100, generations=50))
    exp2.name = "Premium NSGA-II"
    exp2.run(repeat=5)

    def plot_triple(h1, h2, title_suffix, y_max_raw=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        v1_raw, v2_raw = h1[0].values[-1,:].mean(), h2[0].values[-1,:].mean()
        v1_rel, v2_rel = h1[1].values[-1,:].mean(), h2[1].values[-1,:].mean()
        v1_abs, v2_abs = h1[2].values[-1,:].mean(), h2[2].values[-1,:].mean()

        y_max = y_max_raw if y_max_raw else max(1.05, v2_raw * 1.05)

        def annotate(ax, val1, val2, limit=None, limit_label=None, limit_color='gray'):
            ax.axhline(val1, color='blue', linestyle=':', alpha=0.5)
            ax.text(0.5, val1 - y_max*0.01, f"v1={val1:.3f}", color='blue', fontsize=8, va='top')
            ax.axhline(val2, color='darkorange', linestyle=':', alpha=0.5)
            ax.text(0.5, val2 - y_max*0.01, f"v2={val2:.3f}", color='darkorange', fontsize=8, va='top')
            if limit:
                ax.axhline(limit, color=limit_color, linestyle='--', alpha=0.7, label=limit_label)
                ax.legend(loc='lower right', fontsize=8)
            ax.set_ylim([0, y_max])

        # 1. RAW
        mb.view.perf_history(h1[0], h2[0], ax=ax1, title=f"1. Raw\n{title_suffix}", ylabel="Volume")
        annotate(ax1, v1_raw, v2_raw)

        # 2. RELATIVE
        mb.view.perf_history(h1[1], h2[1], ax=ax2, title=f"2. Relative\n{title_suffix}", ylabel="Efficiency")
        annotate(ax2, v1_rel, v2_rel, limit=1.0, limit_label="Session Max")

        # 3. ABSOLUTE
        mb.view.perf_history(h1[2], h2[2], ax=ax3, title=f"3. Absolute\n{title_suffix}", ylabel="Optimality")
        annotate(ax3, v1_abs, v2_abs, limit=1.0, limit_label="Ground Truth", limit_color='gold')
        
        plt.tight_layout()
        plt.show()

    # --- PHASE 1: INDIVIDUAL PERSPECTIVE (joint=False) ---
    print("\n[Phase 1] Calculating Individual Metrics (joint=False)...")
    print("Normalización individual: Each algorithm is evaluated against its local nadir.")
    print("Scale Mode Details:")
    print(" - 'raw': Absolute volume dominated (sensitive to objective ranges).")
    print(" - 'relative': Efficiency normalized [0, 1] against the algorithm's own worst point.")
    print(" - 'absolute': Optimality reach normalized [0, 1] against the Ground Truth.")
    h1_ind = [
        mb.metrics.hv(exp1, scale='raw', joint=False),
        mb.metrics.hv(exp1, scale='relative', joint=False),
        mb.metrics.hv(exp1, scale='absolute', joint=False)
    ]
    h2_ind = [
        mb.metrics.hv(exp2, scale='raw', joint=False),
        mb.metrics.hv(exp2, scale='relative', joint=False),
        mb.metrics.hv(exp2, scale='absolute', joint=False)
    ]
    plot_triple(h1_ind, h2_ind, "[Individual Perspective]")

    # --- PHASE 2: JOINT PERSPECTIVE (joint=True) ---
    print("\n[Phase 2] Calculating Joint Metrics (joint=True)...")
    print("Default Behavior: By default, MoeaBench uses a global Bounding Box (BBox)")
    print("to evaluate all algorithms in a common metric grid. This can be")
    print("manually set via 'ref' or auto-calculated from all involved experiments.")
    print("\nThe 'joint' parameter specifically controls auto-normalization:")
    print(" - joint=True (Default): All sets share a global BBox for a fair comparison.")
    print(" - joint=False: Disables auto-normalization; local BBoxes are used instead.")
    ref = [exp1, exp2]
    h1_jnt = [
        mb.metrics.hv(exp1, ref=ref, scale='raw', joint=True),
        mb.metrics.hv(exp1, ref=ref, scale='relative', joint=True),
        mb.metrics.hv(exp1, ref=ref, scale='absolute', joint=True)
    ]
    h2_jnt = [
        mb.metrics.hv(exp2, ref=ref, scale='raw', joint=True),
        mb.metrics.hv(exp2, ref=ref, scale='relative', joint=True),
        mb.metrics.hv(exp2, ref=ref, scale='absolute', joint=True)
    ]
    # Reuse y_max from the Premium's raw individual volume to see the shift
    y_max_jnt = h2_ind[0].values[-1,:].mean() * 1.1
    plot_triple(h1_jnt, h2_jnt, "[Joint Perspective]", y_max_raw=y_max_jnt)

    print("\nObservation:")
    print("In the Individual Perspective, the Baseline (v1) appears highly competitive.")
    print("In the Joint Perspective, the Baseline 'shrinks' as the Bounding Box expands")
    print("to accommodate the superior solutions found by the Premium (v2) configuration.")

if __name__ == "__main__":
    main()
