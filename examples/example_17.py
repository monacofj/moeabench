#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 17: Convergence, Proximity, and Distributional Correlation Analysis
------------------------------------------------------------------
This example explores the different ways to measure "Quality" in MOAs.

It demonstrates:
1. Distance-based metrics: IGD (Inverted Generational Distance), GD (Generational Distance).
2. Distribution-based metrics: EMD (Earth Mover's Distance/Wasserstein).
3. Metric Correlation: plot_matrix to see if they agree or diverge.
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt

def main():
    print(f"MoeaBench v{mb.system.version()}")
    print("--- Example 17: Convergence, Proximity, and Distributional Correlation Analysis")

    # 1. Setup: A standard problem (DTLZ2, 2 Objectives for easier visualization)
    mop = mb.mops.DTLZ2(M=2)
    
    # We use a single NSGA-II experiment
    exp = mb.experiment()
    exp.name = "Metric Analysis Study"
    exp.mop = mop
    exp.moea = mb.moeas.NSGA2(population=100, generations=50)

    # 2. Execution
    print(f"Running {exp.name}...")
    exp.run(repeat=1) # Single run for metric trajectory analysis

    # 3. Calculating the Portfolio
    # We calculate multiple metrics for the same run to see their agreement
    print("\nCalculating metrics (IGD, GD, EMD)...")
    
    # Distance to GT: Convergence & Proximity
    res_igd = mb.metrics.igd(exp)
    res_gd = mb.metrics.gd(exp)
    
    # Distribution matching: Topological Integrity
    # EMD measures how the "mass" of the population is distributed compared to the GT.
    res_emd = mb.metrics.emd(exp)

    # 4. Metric Correlation Analysis (The plot_matrix)
    # Question: "Do these metrics tell the same story?"
    # The plot_matrix shows the correlation between all calculated metrics.
    print("\nPlotting Metric Correlation Matrix...")
    mb.metrics.plot_matrix(res_igd, res_gd, res_emd, 
                          title="Metric Correlation Portfolio: Do they agree?")

    # 5. Comparing Trajectories
    # We can plot them over time to see different "Maturity" profiles
    print("\nComparing Metric Trajectories...")
    mb.view.perf_history(res_igd, res_gd, res_emd,
                         title="Metric Trajectories: Distances vs Distribution")

    print("\nMetric Portfolio Analysis completed.")
    plt.show()

if __name__ == "__main__":
    main()

# --- Scientific Insight ---
#
# 1. IGD vs GD: IGD (Inverted) measures both proximity and coverage. 
#    A low IGD means the search is close to the front AND well distributed. 
#    GD only measures proximity.
#
# 2. EMD (Earth Mover's Distance): This is more robust than IGD for 
#    "Degenerate" fronts. While IGD might be low if many points are 
#    "somewhat" close, EMD will stay high if the algorithm has "blind spots" 
#    (unfilled regions of the manifold).
#
# 3. Correlation: In the early generations, metrics tend to be highly 
#    correlated (everything is improving). In the final stages, they often 
#    diverge, revealing subtle search pathologies.
