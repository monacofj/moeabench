#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 15: Triple-Mode Hypervolume (Raw, Relative, Absolute)
-------------------------------------------------------------
This example demonstrates the three scaling perspectives in MoeaBench:
1. 'raw': Physical volume conquered (H_raw). Invariant to neighbors.
2. 'relative': Competitive efficiency (H_rel). Scaled by session best.
3. 'absolute': Theoretical optimality (H_abs). Scaled by Ground Truth.

In this example, we compare a Baseline (small budget) vs a Premium (high budget)
configuration and observe their behavior across different normalizations.
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt
import numpy as np

def main():
    print(f"MoeaBench v{mb.system.version()}")
    print("Example 15: The Triple-Mode Perspective")
    print("=======================================\n")

    # 1. Setup the Problem & Calibration
    dtlz2 = mb.mops.DTLZ2(n_var=7, n_obj=3)
    
    # Absolute mode requires calibration data (the dense Ground Truth)
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

    # 2. THE THREE PERSPECTIVES
    print("\n[Phase 1] Calculating Metrics...")
    global_ref = [exp1, exp2]
    
    # A) Raw: Physical volume conquer (H_raw)
    hv1_raw = mb.metrics.hv(exp1, ref=global_ref, scale='raw')
    hv2_raw = mb.metrics.hv(exp2, ref=global_ref, scale='raw')

    # B) Relative: Competitive ranking (H_rel, relative to session best)
    hv1_rel = mb.metrics.hv(exp1, ref=global_ref, scale='relative')
    hv2_rel = mb.metrics.hv(exp2, ref=global_ref, scale='relative')

    # C) Absolute: Theoretical optimality (H_abs, relative to GT)
    hv1_abs = mb.metrics.hv(exp1, scale='absolute')
    hv2_abs = mb.metrics.hv(exp2, scale='absolute')

    print("\n--- RESULTS AT FINAL GENERATION ---")
    print(f"-> Raw [Invariant physical volume]:")
    print(f"     exp1: {hv1_raw.values[-1,:].mean():.4f}")
    print(f"     exp2: {hv2_raw.values[-1,:].mean():.4f}")
    
    print(f"-> Relative [Ranked against session best]:")
    print(f"     exp1: {hv1_rel.values[-1,:].mean():.4f}")
    print(f"     exp2: {hv2_rel.values[-1,:].mean():.4f} (Session Winner)")
    
    print(f"-> Absolute [Theoretical optimality (vs GT)]:")
    print(f"     exp1: {hv1_abs.values[-1,:].mean():.4f}")
    print(f"     exp2: {hv2_abs.values[-1,:].mean():.4f}")

    # 3. Visual Comparison
    print("\nGenerating comparative plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Extract final values for annotations
    v1_raw, v2_raw = hv1_raw.values[-1,:].mean(), hv2_raw.values[-1,:].mean()
    v1_rel, v2_rel = hv1_rel.values[-1,:].mean(), hv2_rel.values[-1,:].mean()
    v1_abs, v2_abs = hv1_abs.values[-1,:].mean(), hv2_abs.values[-1,:].mean()

    # Unified Y-axis based on Raw
    y_max = max(1.05, v2_raw * 1.05)
    
    # Helper function to add annotated horizontal lines
    def annotate_lines(ax, val1, val2):
        # Val 1: Blue
        ax.axhline(val1, color='blue', linestyle=':', alpha=0.5)
        ax.text(0.5, val1 - y_max*0.01, f"hv1 = {val1:.3f}", 
                color='blue', fontsize=8, va='top', ha='left')
        
        # Val 2: Orange
        ax.axhline(val2, color='darkorange', linestyle=':', alpha=0.5)
        ax.text(0.5, val2 - y_max*0.01, f"hv2 = {val2:.3f}", 
                color='darkorange', fontsize=8, va='top', ha='left')
        
        ax.set_ylim([0, y_max])
    
    # Plot 1: RAW
    # No 'labels' passed -> Engine will use standard 'Name' format for history
    mb.view.perf_history(hv1_raw, hv2_raw, ax=ax1, 
                         title="1. Physical (Raw)\n[Invariant Absolute Volume]", 
                         ylabel="Volumetric Units")
    annotate_lines(ax1, v1_raw, v2_raw)

    # Plot 2: RELATIVE
    mb.view.perf_history(hv1_rel, hv2_rel, ax=ax2, 
                         title="2. Competitive (Relative)\n[Forced 1.0 Ceiling]", 
                         ylabel="Efficiency Ratio")
    annotate_lines(ax2, v1_rel, v2_rel)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label="Session Winner (1.0)")
    ax2.legend(loc='lower right', fontsize=8)

    # Plot 3: ABSOLUTE
    mb.view.perf_history(hv1_abs, hv2_abs, ax=ax3, 
                         title="3. Theoretical (Absolute)\n[Anchored to GT]", 
                         ylabel="Optimality Ratio")
    annotate_lines(ax3, v1_abs, v2_abs)
    ax3.axhline(1.0, color='gold', linestyle='--', alpha=1.0, label="Ground Truth (1.0)")
    ax3.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
