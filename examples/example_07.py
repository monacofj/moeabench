#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 07: Empirical Attainment Functions (EAF)
----------------------------------------------
This example demonstrates how to use attainment surfaces to analyze 
the statistical distribution of Pareto fronts across multiple runs.
It visualizes the reliability bands of a search process.
"""

import mb_path
from MoeaBench import mb

def main():
    print(f"Version: {mb.system.version()}")
    print("--- Empirical Attainment Workshop ---\n")
    
    # 1. Setup: 2D problem for clear staircase visualization
    mop1 = mb.mops.DTLZ2(M=2)
    repeats = 10 
    
    exp1 = mb.experiment()
    exp1.name = "NSGA-II"
    exp1.mop = mop1
    exp1.moea = mb.moeas.NSGA2deap(population=100, generations=100)

    print(f"Executing {exp1.name} ({repeats} runs)...")
    exp1.run(repeat=repeats)
    
    # 2. Attainment Surfaces (Reliability Bands)
    print("Calculating Attainment Surfaces (Optimistic, Median, Pessimistic)...")
    
    # surf1 contains:
    #             .level           attainment level (0.5 = median)
    #             .objectives      surface coordinates
    #             .volume()        attained volume
    surf1 = mb.stats.attainment(exp1, level=0.1) # Best 10%
    surf1.name = f"{exp1.name} (10% Best)"
    
    surf2 = mb.stats.attainment(exp1, level=0.5) # Median
    surf2.name = f"{exp1.name} (Median)"
    
    surf3 = mb.stats.attainment(exp1, level=0.9) # Worst 10%
    surf3.name = f"{exp1.name} (90% Worst)"
    
    # Visualize the "Search Corridor" (Spatial Perspective)
    print("Plotting reliability band...")
    mb.view.spaceplot(surf1, surf2, surf3, title="NSGA-II Search Corridor")
    
    # 3. Comparative Attainment
    print(f"\nComparing with SPEA2...")
    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mop1
    exp2.moea = mb.moeas.SPEA2(population=50, generations=100)
    exp2.run(repeat=repeats)
    
    # res1 contains:
    #             .surf1           attainment surface of exp1
    #             .surf2           attainment surface of exp2
    #             .volume_diff     difference in attained volumes
    #             .report()        narrative summary
    res1 = mb.stats.attainment_diff(exp1, exp2, level=0.5)
    print(res1.report())
    
    # The diff object is iterable, returning (surf1, surf2) for plotting (Spatial Perspective)
    mb.view.spaceplot(*res1, title="Median Attainment: NSGA-II vs SPEA2")

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# While Hypervolume gives a single number, Attainment Surfaces show *where* 
# the algorithm succeeds or fails in objective space.
#
# The reliability band (10%, 50%, 90%) reveals how much you can trust your 
# results. A wide band means high variability; a narrow band means the 
# algorithm is very consistent.
#
# In the comparison, 'attainment_diff' highlights the regions where one 
# algorithm dominates the other's typical performance.
