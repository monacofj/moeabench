#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 08: Advanced Population Diagnostics Workshop
--------------------------------------------------
This example demonstrates the complete stratification suite, visualizing
the "Population Geology" (Dominance Structure) of different search algorithms.

It focuses on the new `strat_caste` visualizer (v0.8.0), exploring:
1. Stochastic Robustness (Collective Mode)
2. Internal Diversity (Individual Mode)
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt

def main():
    print(f"Version: {mb.system.version()}")
    print("--- Advanced Diagnostics Workshop ---\n")

    # 1. Setup: Standard 3D problem (DTLZ2)
    # NSGA-III vs SPEA2: A classic head-to-head comparison
    mop = mb.mops.DTLZ2(M=3)
    repeats = 10 

    exp1 = mb.experiment()
    exp1.name = "NSGA-III"
    exp1.mop = mop
    exp1.moea = mb.moeas.NSGA3(population=100, generations=50, seed=1)

    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mop
    exp2.moea = mb.moeas.SPEA2(population=100, generations=50, seed=42)

    print(f"Running experiments ({repeats} repeats each)...")
    exp1.run(repeat=repeats)
    exp2.run(repeat=repeats)

    # 2. Analysis: Snapshot at Early Search (Gen 10)
    SNAPSHOT_GEN = 10
    print(f"\n--- Diagnostic Snapshot at Gen {SNAPSHOT_GEN} ---")
    
    strat1 = mb.stats.strata(exp1, gen=SNAPSHOT_GEN)
    strat2 = mb.stats.strata(exp2, gen=SNAPSHOT_GEN)

    # 3. Visual Workshop
    print("\nGenerating visual profiles... (Check the plots)")

    # A. Micro-Analysis: Population Diversity (Individual Mode)
    # Question: "What is the Per Capita merit of the citizens?"
    #
    # Systemic and Biological Interpretation:
    # ----------------------------------------
    # This mode measures Internal Diversity. Each data point represents the quality 
    # of a single solution within the population.
    #
    # 1. Population Density (n): Mean count of solutions occupying the rank. 
    #    Example: n=94 indicates 94% of the population is in the first rank, 
    #    denoting extremely high Selection Pressure.
    # 2. Median Merit (q): Bold numerical value representing individual quality.
    #    It establishes the center of gravity of the front.
    # 3. The Box (Quartiles): Spans from Q1 to Q3.
    #    Measures uniformity. A tall box indicates high diversity (containing 
    #    both high-performance architects and marginal successors).
    # 4. The Whiskers: Extend to 1.5 x IQR. 
    #    Mark the boundaries of "Normal" solutions. Points beyond are rare 
    #    Mutants or Outliers—solutions so unique they break the distribution.
    ax_ind = mb.view.strat_caste(strat1, strat2, mode='individual', 
                 title=f"Individual Perspective: Solution Merit - Gen {SNAPSHOT_GEN}")
    ax_ind.figure.savefig("strat_caste_individual.png", dpi=300, bbox_inches='tight')

    # B. Macro-Analysis: Stochastic Robustness (Collective Mode)
    # Scientific Inquiry: "What is the aggregate performance distribution across runs?"
    #
    # Statistical Interpretation:
    # - Observation Point: Total rank quality (e.g., Hypervolume contribution) 
    #   per execution.
    # - Box Height (Dispersion): Direct indicator of stochastic determinism. 
    #   Minimal dispersion suggests high reliability across trials.
    # - Outliers: Detect rare convergence failures or significant performance 
    #   deviations within the sample.
    ax_coll = mb.view.strat_caste(strat1, strat2, mode='collective', 
                 title=f"Macro View: Stochastic Robustness - Gen {SNAPSHOT_GEN}")
    ax_coll.figure.savefig("strat_caste_collective.png", dpi=300, bbox_inches='tight')
    
    # C. Competitive View (The Tier Duel)
    print(f"\n--- Competitive Tier Duel (F1 Pole/Gap) ---")
    res_tier = mb.stats.tier(exp1, exp2, gen=SNAPSHOT_GEN)
    print(res_tier.report())
    ax_tier = mb.view.strat_tiers(res_tier, title="Competitive Perspective: Tier Duel")
    ax_tier.figure.savefig("strat_tiers_duel.png", dpi=300, bbox_inches='tight')
    print("Saved 'strat_tiers_duel.png'")

    print("\nVisual profiles generated. Check the PNG files or the window if available.")
    plt.show(block=True)

if __name__ == "__main__":
    main()
