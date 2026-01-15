#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 01: Basic Experiment Cycle
----------------------------------
This example demonstrates the fundamental workflow of MoeaBench:
defining a problem (MOP), choosing an algorithm (MOEA), 
running the experiment, and visualizing the Pareto Front.
"""

import mb_path
from MoeaBench import mb

def main():
    # 1. Setup: Define a 3-objective DTLZ2 problem and NSGA-III
    exp1 = mb.experiment()
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=50, generations=50)

    # 2. Execution: Run the optimization
    print("Running experiment...")
    exp1.run()
    
    # 3. Visualization: Inspect the found Pareto Front
    # We plot the final non-dominated solutions in objective space.
    print("Plotting results...")
    mb.spaceplot(exp1.front(), title="NSGA-III Final Front")

    # Calculate and plot Hypervolume convergence over generations
    # hv1 contains:
    #             MetricMatrix    historical hypervolume values
    hv1 = mb.hv(exp1)
    mb.timeplot(hv1, title="Hypervolume Convergence")

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# In this first example, we see how MoeaBench encapsulates the optimization cycle.
# The 'spaceplot' shows the distribution of non-dominated solutions found.
# Even with a small population (50), NSGA-III begins to approximate the 
# spherical nature of the DTLZ2 front.
#
# The 'timeplot' reveals the convergence behavior. We expect a steep initial 
# curve that plateaus once the algorithm reaches the true Pareto surface.
