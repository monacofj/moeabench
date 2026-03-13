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
1. spread: Boxplots with A12 Win Probability & P-value annotations.
2. bands: Reliability envelopes based on Empirical Attainment Functions (EAF).
"""

import mb_path
import moeabench as mb
import matplotlib.pyplot as plt
from moeabench.core.display import show_matplotlib

def main():
    mb.system.version()

    # 1. Setup: A competitive comparison on DTLZ2 (3 Objectives)
    mop3 = mb.mops.DTLZ2(M=3)
    
    exp1_3D = mb.experiment()
    exp1_3D.name = "NSGA-II (3D)"
    exp1_3D.mop = mop3
    exp1_3D.moea = mb.moeas.NSGA2(population=100, generations=100)
    
    exp2_3D = mb.experiment()
    exp2_3D.name = "MOEA/D (3D)"
    exp2_3D.mop = mop3
    exp2_3D.moea = mb.moeas.MOEAD(population=100, generations=100)

    REPEATS = 10
    exp1_3D.run(repeat=REPEATS)
    exp2_3D.run(repeat=REPEATS)

    mb.view.bands(exp1_3D, exp2_3D, levels=[0.1, 0.5, 0.9], style='step', title="3D Search Corridors (STEP)")

    mb.view.bands(exp1_3D, exp2_3D, levels=[0.1, 0.5, 0.9], style='spline', title="3D Search Corridors (SPLINE)")


    # 2. Setup: 2-Objective Analysis for Band Fill
    mop2 = mb.mops.DTLZ2(M=2)
    
    exp1_2D = mb.experiment()
    exp1_2D.name = "NSGA-II (2D)"
    exp1_2D.mop = mop2
    exp1_2D.moea = mb.moeas.NSGA2(population=100, generations=100)
    
    exp2_2D = mb.experiment()
    exp2_2D.name = "MOEA/D (2D)"
    exp2_2D.mop = mop2
    exp2_2D.moea = mb.moeas.MOEAD(population=100, generations=100)

    exp1_2D.run(repeat=REPEATS)
    exp2_2D.run(repeat=REPEATS)

    mb.view.bands(exp1_2D, exp2_2D, levels=[0.1, 0.5, 0.9], style='fill', title="2D Search Corridors (FILL)")

    mb.view.bands(exp1_2D, exp2_2D, levels=[0.1, 0.5, 0.9], style='spline', title="2D Search Corridors (SPLINE)")
                       
    mb.view.bands(exp1_2D, exp2_2D, levels=[0.1, 0.5, 0.9], style='step', title="2D Search Corridors (STEP)")

    show_matplotlib()

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# 1. spread: This is the definitive way to report performance. 
#    - The Boxplot shows the distribution of final Hypervolume.
#    - The 'A12' value (Vargha-Delaney) is the "Effect Size": 1.0 means A always wins, 
#      0.5 means they are perfectly tied.
#    - The 'p-value' provides the statistical significance.
#
# 2. bands: Visualizes the "Search Corridor".
#    - Multi-objective solvers are stochastic. This plot shows that despite having 
#      similar mean HV, one algorithm might be much more consistent spatially 
#      than the other (narrower bands).
