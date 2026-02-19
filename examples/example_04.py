#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 04: Multi-run Reliability and Stability
-----------------------------------------------
This example demonstrates how to handle stochastic variability by running 
an experiment multiple times (multi-run) and visualizing the statistical 
stability of the convergence.
"""

import mb_path
from MoeaBench import mb

def main():
    print(f"MoeaBench v{mb.system.version()}")
    # 1. Setup: Multiple runs for NSGA-III
    exp1 = mb.experiment()
    exp1.name = "NSGA-III (5 runs)"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=50, generations=50)

    # 2. Execution: Run 5 times with different seeds
    # Running multiple times allows us to calculate confidence intervals.
    print(f"Executing {exp1.name}...")
    exp1.run(repeat=5)
    
    # 3. Aggregated Convergence (Performance Domain)
    # hv1 contains the historical hypervolume for ALL runs.
    hv1 = mb.metrics.hv(exp1)

    # NEW: Statistical summary of multi-run performance
    hv1.report_show()

    # The perf_history automatically computes mean and standard deviation
    print("Plotting statistical convergence...")
    mb.view.perf_history(hv1, title="Stability Analysis (5-run HV)")

    # 4. Aggregated Quality (Topographic Domain)
    # The 'front()' method provides the combined non-dominated solutions 
    # considering the discovery of all runs (The Superfront).
    print("Plotting Superfront...")
    mb.view.topo_shape(exp1.front(), title="Combined Global Front (Superfront)")

    # 5. Stability Inspection (Topographic Domain)
    # We can also plot each run's final front independently.
    print("Comparing individual run stability...")
    mb.view.topo_shape(*exp1.all_fronts(), title="Individual Run Fronts")

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# Multi-objective optimization is inherently stochastic. A single run might be 
# lucky or unlucky. By running multiple times (repeat=5), we get a 
# "silhouette" of the algorithm's performance.
#
# The 'perf_history' dispersion shadow (mean +/- std) shows the reliability. 
# A thin shadow indicates high consistency.
#
# The 'superfront' is the definitive result for the user: it's the best 
# knowledge we have about the problem after several independent search attempts.
