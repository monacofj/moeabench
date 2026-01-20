# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 11: Scientific Domains Taxonomy (The "Scientific Tridade")
------------------------------------------------------------------
This example demonstrates the MoeaBench v0.7.0 taxonomy, organizing 
analysis into three domains: Topography, Performance, and Stratification.
"""

import mb_path
from MoeaBench import mb

def main():
    print(f"MoeaBench Version: {mb.system.version()}")
    print("-" * 30)

    # 1. Setup Study
    exp1 = mb.experiment()
    exp1.name = "NSGA3 (Standard)"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=100)
    
    exp2 = mb.experiment()
    exp2.name = "MOEAD (Rival)"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.MOEAD(population=100)

    # 2. Execute Research (Multi-run)
    print("Executing multi-run experiments...")
    exp1.run(repeat=5, generations=100)
    exp2.run(repeat=5, generations=100)

    # --- DOMAIN 1: TOPOGRAPHY (topo_*) ---
    # Geographic Focus: Where are the solutions?
    print("\nVisualizing Topography...")
    
    # 1.1 Structural Shape
    mb.view.topo_shape(exp1.front(), exp1.optimal_front(), title="Topographic Shape (NSGA3)")
    
    # 1.2 Search Corridors (Reliability Bands)
    mb.view.topo_bands(exp1, levels=[0.5, 0.9], title="Search Reliability Bands (NSGA3)")
    
    # 1.3 Topologic Gap (Coverage Difference)
    mb.view.topo_gap(exp1, exp2, title="Topologic Gap: NSGA3 vs MOEAD")

    # --- DOMAIN 2: PERFORMANCE (perf_*) ---
    # Utility Focus: How well did they perform?
    print("\nVisualizing Performance...")
    
    # 2.1 Performance History (Convergence)
    mb.view.perf_history(exp1, exp2, title="Convergence History (Hypervolume)")
    
    # 2.2 Performance Contrast (Statistical Spread)
    mb.view.perf_spread(exp1, exp2, title="Performance Contrast (v0.7.0)")

    # --- DOMAIN 3: STRATIFICATION (strat_*) ---
    # Geological Focus: Internal population structure
    print("\nVisualizing Stratification...")
    
    # 3.1 Structural Ranks
    mb.view.strat_ranks(exp1, exp2, title="Selection Pressure (Rank Distribution)")
    
    # 3.2 Competitive Tier Duel
    mb.view.strat_tiers(exp1, exp2, title="Competitive Tier Duel")

    print("\nScientific domains showcase completed.")

if __name__ == "__main__":
    main()
