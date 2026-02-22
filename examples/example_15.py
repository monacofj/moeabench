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

def main():
    print(f"MoeaBench v{mb.system.version()}")
    print("Example 15: Dual-Mode Hypervolume (Raw vs Ratio)")
    print("================================================\n")

    # 1. Setup the Problem
    dtlz2 = mb.mops.DTLZ2(n_var=12, n_obj=3)
    
    # Corrected usage: Using the experiment object directly
    print("Running Experiment 1 (Small Population)...")
    exp1 = mb.experiment(dtlz2, mb.moeas.NSGA2(population=20, generations=50))
    exp1.name = "NSGA-II (Pop: 20)"
    exp1.run(repeat=5)
    
    print("Running Experiment 2 (Large Population)...")
    exp2 = mb.experiment(dtlz2, mb.moeas.NSGA2(population=100, generations=50))
    exp2.name = "NSGA-II (Pop: 100)"
    exp2.run(repeat=5)

    # 2. Extracting the Metrics
    print("\nCalculating Hypervolume in both modes...")
    
    # We use both experiments as reference to ensure a shared Bounding Box.
    # This is crucial for comparing 'raw' volumes fairly.
    refs = [exp1, exp2]
    
    # Mode 1: Raw (Now the Default). Returns actual geometric volume.
    hv_raw1 = mb.metrics.hv(exp1, ref=refs) 
    hv_raw2 = mb.metrics.hv(exp2, ref=refs)
    
    # Mode 2: Ratio. Scales results so the best algorithm reaches 1.0.
    hv_ratio1 = mb.metrics.hv(exp1, ref=refs, scale='ratio')
    hv_ratio2 = mb.metrics.hv(exp2, ref=refs, scale='ratio')

    # 3. Textual Reporting
    print("\n--- RATIO MODE REPORT (Competitive Efficiency) ---")
    print("Notice how the best algorithm defines the 1.0 ceiling.")
    print(hv_ratio1.report()) 
    print("\n")
    print(hv_ratio2.report()) 

    print("\n--- RAW MODE REPORT (Absolute Physical Volume) ---")
    print("Notice the raw numerical values reflecting actual space dominated.")
    print(hv_raw1.report())   
    print("\n")
    print(hv_raw2.report())   

    # 4. Visual Comparison
    print("\nGenerating comparative plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    mb.view.perf_history(hv_ratio1, hv_ratio2, ax=ax1, 
                         title="Ratio Mode (Rank/Efficiency)", 
                         ylabel="Hypervolume (Ratio to Max)")
    
    mb.view.perf_history(hv_raw1, hv_raw2, ax=ax2, 
                         title="Raw Mode (Physical Space)", 
                         ylabel="Absolute Hypervolume")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
