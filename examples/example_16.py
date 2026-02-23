#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 16: Comparative Performance Contrast and Stochastic Reliability
--------------------------------------------------------------------------
This example demonstrates how to perform a rigorous head-to-head comparison 
between two algorithms suitable for a scientific publication.

It features:
1. perf_spread: Boxplots with A12 Win Probability & P-value annotations.
2. topo_bands: Reliability envelopes based on Empirical Attainment Functions (EAF).
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt

def main():
    print(f"MoeaBench v{mb.system.version()}")
    print("--- Example 16: Comparative Performance Contrast and Stochastic Reliability")

    # 1. Setup: A competitive comparison on DTLZ2 (3 Objectives)
    mop = mb.mops.DTLZ2(M=3)
    
    # We compare NSGA-II vs MOEA/D
    exp1 = mb.experiment()
    exp1.name = "NSGA-II"
    exp1.mop = mop
    exp1.moea = mb.moeas.NSGA2(population=100, generations=100)

    exp2 = mb.experiment()
    exp2.name = "MOEA/D"
    exp2.mop = mop
    exp2.moea = mb.moeas.MOEAD(population=100, generations=100)

    # 2. Execution: Multiple runs are required for statistical rigor
    REPEATS = 15
    print(f"Running {exp1.name} ({REPEATS} times)...")
    exp1.run(repeat=REPEATS)
    
    print(f"Running {exp2.name} ({REPEATS} times)...")
    exp2.run(repeat=REPEATS)

    # 3. Performance Contrast (The Publication Chart)
    # Questions: 
    # - "Is the difference statistically significant (p < 0.05)?"
    # - "What is the probability that Alg A outperforms Alg B (A12)?"
    print("\nPlotting Performance Contrast (perf_spread)...")
    mb.view.perf_spread(exp1, exp2, title="Metric Contrast: NSGA-II vs MOEA/D")

    # 4. Stochastic Reliability (EAF Bands)
    # Question: "What is the search corridor reached by at least 50% of the runs?"
    # The 'topo_bands' highlights the median attainment and the 10%-90% reliability envelope.
    print("\nPlotting Reliability Envelopes (topo_bands)...")
    mb.view.topo_bands(exp1, exp2, 
                       levels=[0.1, 0.5, 0.9], 
                       title="Search Corridors: Stochastic Reliability (EAF Bands)")

    print("\nScientific plots generated. Review the figures to see A12 and EAF envelopes.")
    plt.show()

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# 1. perf_spread: This is the definitive way to report performance. 
#    - The Boxplot shows the distribution of final Hypervolume.
#    - The 'A12' value (Vargha-Delaney) is the "Effect Size": 1.0 means A always wins, 
#      0.5 means they are perfectly tied.
#    - The 'p-value' provides the statistical significance.
#
# 2. topo_bands: Visualizes the "Search Corridor".
#    - Multi-objective solvers are stochastic. This plot shows that despite having 
#      similar mean HV, one algorithm might be much more consistent spatially 
#      than the other (narrower bands).
