#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 07: Empirical Attainment Functions (EAF)
----------------------------------------------
This example demonstrates how to use attainment surfaces to analyze 
the statistical distribution of Pareto fronts across multiple runs.
It visualizes the "Expected" front (50%) along with the 
"Optimistic" (5%) and "Pessimistic" (95%) boundaries.
"""

import mb_path
import MoeaBench as mb

import numpy as np

def run_example():
    print("--- Empirical Attainment Functions Example ---")
    
    # Setup Problem: 2D for clear staircase visualization
    mop = mb.mops.DTLZ2(M=2)
    pop_size = 52
    gens = 100
    repeats = 10  # Multiple runs to build the distribution
    
    print(f"Running {repeats} repetitions for NSGA2...")
    
    # 1. Run NSGA2
    exp1 = mb.experiment()
    exp1.name = "NSGA2"
    exp1.mop = mop
    exp1.moea = mb.moeas.NSGA2deap(population=pop_size, generations=gens)
    exp1.run(repeat=repeats, workers=0) # workers=0 uses half CPUs
    
    # 2. Calculate Attainment Surfaces
    print("Calculating Attainment Surfaces (5%, 50%, 95%)...")
    
    # Median attainment (Expected performance)
    median = mb.stats.attainment(exp1, level=0.5)
    median.name = "NSGA2 (Median 50%)"
    
    # Optimistic attainment (Best 5% cases)
    best = mb.stats.attainment(exp1, level=0.1)
    best.name = "NSGA2 (Best 10%)"
    
    # Pessimistic attainment (Worst 95% cases)
    worst = mb.stats.attainment(exp1, level=0.9)
    worst.name = "NSGA2 (Worst 90%)"
    
    # 3. Visualize the "Reliability Band"
    # This shows the area where the algorithm's performance usually falls.
    print("Visualizing Reliability Band...")
    mb.spaceplot(best, median, worst, title="NSGA2 Attainment Reliability")
    
    # 4. Comparative Attainment
    print(f"\nComparing with SPEA2...")
    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mop
    exp2.moea = mb.moeas.SPEA2(population=pop_size, generations=gens)
    exp2.run(repeat=repeats, workers=0)
    
    # We can use the attainment_diff helper
    print("Calculating and visualizing attainment difference (Median)...")
    s1, s2 = mb.stats.attainment_diff(exp1, exp2, level=0.5)
    
    mb.spaceplot(s1, s2, title="Comparison: NSGA2 vs SPEA2 (Median)")

if __name__ == "__main__":
    run_example()
