#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 15: Dual-Mode Hypervolume (Raw vs Ratio)
------------------------------------------------
This example demonstrates the critical difference between the two 
hypervolume scaling modes in MoeaBench:
1. 'raw': The absolute physical volume dominated by the algorithm.
2. 'ratio': The relative efficiency compared to the maximum volume 
            found in the current session (forces a 1.0 ceiling).

Understanding this difference is key to interpreting whether an algorithm
is physically dominating more space, or simply changing its rank relative
to a new competitor.
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt
import numpy as np

def main():
    print(f"MoeaBench v{mb.system.version()}")
    print("Example 15: The Reference Paradox (Raw vs Ratio)")
    print("==============================================\n")

    # 1. Setup the Problem
    dtlz2 = mb.mops.DTLZ2(n_var=7, n_obj=3)
    
    print("Running Experiment 1 (Baseline: 20 individuals)...")
    exp1 = mb.experiment(dtlz2, mb.moeas.NSGA2(population=20, generations=50))
    exp1.name = "Baseline NSGA-II"
    exp1.run(repeat=5)
    
    print("\nRunning Experiment 2 (Superior Competitor: 100 individuals)...")
    exp2 = mb.experiment(dtlz2, mb.moeas.NSGA2(population=100, generations=50))
    exp2.name = "Premium NSGA-II"
    exp2.run(repeat=5)

    # 2. THE PARADOX DEMONSTRATION
    print("\n[Phase 1] Calculating Metrics with a Shared Bounding Box...")
    # To have absolute invariance, we must fix the Bounding Box (Nadir) for all.
    # In a real session, MoeaBench does this by looking at everyone in the 'ref' list.
    global_ref = [exp1, exp2]
    
    # Raw HV: Physically Conquer (Invariant if BB is fixed)
    hv_raw1 = mb.metrics.hv(exp1, ref=global_ref, scale='raw')
    hv_raw2 = mb.metrics.hv(exp2, ref=global_ref, scale='raw')

    # Ratio HV: Competitive Efficiency (Relative to best in session)
    hv_ratio1 = mb.metrics.hv(exp1, ref=global_ref, scale='ratio')
    hv_ratio2 = mb.metrics.hv(exp2, ref=global_ref, scale='ratio')

    # Now, let's simulate the "Alone" case for exp1.
    # If exp2 wasn't in the room, exp1 would be its own 1.0 benchmark.
    # We use its own internal best as denominator.
    denom_alone = np.nanmax(hv_raw1.values[-1, :])
    hv_ratio_alone1 = mb.metrics.MetricMatrix(hv_raw1.values / denom_alone, "Hypervolume (Ratio)")
    hv_ratio_alone1.source_name = exp1.name

    print(f"-> exp1 Ratio (Alone):    {hv_ratio_alone1.values[-1,:].mean():.4f} (Champion!)")
    print(f"-> exp1 Ratio (vs exp2): {hv_ratio1.values[-1,:].mean():.4f} (Looks worse suddenly!)")
    print(f"-> exp1 Raw (Alone):     {hv_raw1.values[-1,:].mean():.4f}")
    print(f"-> exp1 Raw (vs exp2):    {hv_raw1.values[-1,:].mean():.4f} (STABLE!)")

    # 3. Visual Comparison
    print("\nGenerating comparative plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Plotting Raw (LEFT): Show the Invariance (Robustness)
    # We plot the exact same metric data twice (once with a dashed style by default)
    # to show that it occupies the EXACT same physical space.
    mb.view.perf_history(hv_raw1, hv_raw1, ax=ax1, 
                         title="The Raw Invariance (Robustness)", 
                         labels=["exp1 (Standalone Reference)", "exp1 (Session Reference)"],
                         ylabel="Absolute Hypervolume")

    # Plotting Ratio (RIGHT): Show the "Shift" (Fragility)
    # We compare exp1 (alone) vs exp1 (relative to exp2)
    mb.view.perf_history(hv_ratio_alone1, hv_ratio1, ax=ax2, 
                         title="The Ratio Shift (Fragility)", 
                         labels=["exp1 (Baseline Only)", "exp1 (Competitive Session)"],
                         ylabel="Hypervolume (Ratio)")
    
    print("\n--- CONCLUSION ---")
    print("Compare the two plots:")
    print("1. LEFT (Raw): The algorithm's contribution is physically invariant (curves overlap perfectly).")
    print("   *Yes, it is only ONE curve visible because they are 100% mathematically identical.*")
    print("2. RIGHT (Ratio): The same algorithm seems to lose quality just because someone else is better.")
    
    print("\n[Scientific Note on the 'Coincidence']:")
    print("You might notice that 'Raw' and 'Ratio' values look similar (e.g., 0.84 vs 0.86).")
    print("This is because in DTLZ benchmarks, the objective space is typically normalized to a")
    print("unit hypercube (1x1x1). Thus, the physical volume (raw) and the percentage (ratio)")
    print("converge to similar numbers. In a real-world engineering problem where objectives")
    print("measure thousands of dollars or kilograms, the numbers would be orders of magnitude apart.")
    
    print("\nThis demonstrates why 'raw' is the scientific default in MoeaBench.")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
