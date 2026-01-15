#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 06: Statistical Hypothesis Testing
------------------------------------------
This example demonstrates how to perform rigorous statistical comparisons 
between two algorithms using non-parametric tests and effect size measures.
"""

import mb_path
from MoeaBench import mb

def main():
    print("--- Statistical Analysis Workshop ---\n")
    
    # 1. Setup: Compare NSGA-III and SPEA2 with 10 repetitions
    repeats = 10
    pop_size = 100
    gens = 50
    
    exp1 = mb.experiment()
    exp1.name = "NSGA-III"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=pop_size, generations=gens)

    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=pop_size, generations=gens)

    print(f"Running {repeats} repetitions for each algorithm...")
    exp1.run(repeat=repeats)
    exp2.run(repeat=repeats)

    # 2. Statistical Inference
    print("\n--- Inferential Statistics ---")
    
    # res1 contains:
    #             .statistic       test statistic (U)
    #             .p_value         probability of observing the data by chance
    #             .significant     boolean (p < 0.05)
    #             .report()        narrative summary
    res1 = mb.stats.mann_whitney(exp1, exp2)
    print(res1.report())

    # res2 contains:
    #             .statistic       KS distance (D)
    #             .p_value         probability of distribution similarity
    #             .report()        narrative summary
    res2 = mb.stats.ks_test(exp1, exp2)
    print("\n" + res2.report())
        
    # res3 contains:
    #             .value           Vargha-Delaney A12 effect size [0, 1]
    #             .report()        narrative summary
    res3 = mb.stats.a12(exp1, exp2)
    print("\n" + res3.report())

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# Statistical tests avoid the trap of "visual optimization." 
# The Mann-Whitney U test tells us if one algorithm is significantly better 
# than the other based on the median performance.
#
# However, p-values can be misleading with large samples. That's why we 
# include the A12 Effect Size. It tells us the *magnitude* of the difference: 
# an A12 of 0.5 means they are equal; 1.0 means the first always beats the second.
#
# In MoeaBench, these tests are "smart": they automatically handle the 
# extraction of metrics (like Hypervolume) and set a common reference point.
