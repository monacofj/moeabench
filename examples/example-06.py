#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import mb_path # Setup local environment
from MoeaBench import mb
import numpy as np

def main():
    print("--- Statistical Analysis Example ---\n")
    
    # 1. Setup Algorithm Comparison
    # We compare NSGA3 vs SPEA2 on DTLZ2 (3 objectives)
    # We run 10 independent seeds for statistical significance.
    repeats = 10
    pop_size = 100
    gens = 50
    
    print(f"Running {repeats} repetitions for NSGA3 and SPEA2...")
    
    # Experiment 1: NSGA3
    exp1 = mb.experiment()
    exp1.name = "NSGA3"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=pop_size, generations=gens)
    exp1.run(repeat=repeats)
    
    # Experiment 2: SPEA2
    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=pop_size, generations=gens)
    exp2.run(repeat=repeats)

    # 2. Extract Metrics (Hypervolume)
    # mb.hv(exp) returns a matrix (Generations x Repeats)
    # We want the final Hypervolume for each run (last row)
    
    print("\nCalculating metrics...")
    
    hv1 = mb.hv(exp1)
    hv2 = mb.hv(exp2)
    
    #  - .gens(i): Returns distribution across runs at generation i
    #  - .runs(i): Returns trajectory of run i
    # Default is the last generation/run
    
    # We want the distribution at the final generation
    v1 = hv1.gens() # Default -1
    v2 = hv2.gens() # Default -1
    
    # Descriptives
    print(f"\nResults (Hypervolume):")
    print(f"{exp1.name}: Mean={np.mean(v1):.4f}, Std={np.std(v1):.4f}, Median={np.median(v1):.4f}")
    print(f"{exp2.name}: Mean={np.mean(v2):.4f}, Std={np.std(v2):.4f}, Median={np.median(v2):.4f}")
    
    # 3. Statistical Testing
    print("\n--- Statistical Tests ---")
    
    # Mann-Whitney U Test (Significance)
    # Null Hypothesis: The distributions are equal.
    # Result: object with .statistic and .pvalue
    mw_result = mb.stats.mann_whitney(v1, v2)
    p_value = mw_result.pvalue
    
    print(f"Mann-Whitney U P-value: {p_value:.2e}")
    if p_value < 0.05:
        print("  -> Significant Difference (p < 0.05)!")
    else:
        print("  -> No Significant Difference.")
        
    # Vargha-Delaney A12 (Effect Size)
    # Measures probability that 1 > 2.
    # 0.5 = Equal, >0.5 means 1 is better (higher metric), <0.5 means 2 is better.
    # (Assuming higher metric is better, like Hypervolume)
    
    effect_size = mb.stats.a12(v1, v2)
    print(f"Vargha-Delaney A12 (1 vs 2): {effect_size:.4f}")
    
    if effect_size == 0.5:
        print("  -> Magnitude: Negligible")
    elif effect_size > 0.5:
        print(f"  -> {exp1.name} is better than {exp2.name}")
    else:
        print(f"  -> {exp2.name} is better than {exp1.name}")

if __name__ == "__main__":
    main()
