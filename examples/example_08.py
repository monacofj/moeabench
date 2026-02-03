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

    # 1. Setup: Deceptive 3D problem (DTLZ3)
    # NSGA-III vs MOEA/D on a challenging multimodal landscape
    mop = mb.mops.DTLZ3(M=3)
    repeats = 5 

    exp1 = mb.experiment()
    exp1.name = "NSGA-III"
    exp1.mop = mop
    exp1.moea = mb.moeas.NSGA3(population=100, generations=100, seed=1)

    exp2 = mb.experiment()
    exp2.name = "MOEA/D"
    exp2.mop = mop
    exp2.moea = mb.moeas.MOEAD(population=100, generations=100, seed=42)

    print(f"Running experiments ({repeats} repeats each)...")
    exp1.run(repeat=repeats)
    exp2.run(repeat=repeats)

    # 2. Analysis: Snapshot at Mid-Search (Gen 50)
    SNAPSHOT_GEN = 50
    print(f"\n--- Diagnostic Snapshot at Gen {SNAPSHOT_GEN} ---")
    
    strat1 = mb.stats.strata(exp1, gen=SNAPSHOT_GEN)
    strat2 = mb.stats.strata(exp2, gen=SNAPSHOT_GEN)

    # 3. Visual Workshop
    print("\nGenerating visual profiles... (Check the plots)")

    # A. The Macro View: Stochastic Robustness (Collective Mode)
    ax_coll = mb.view.strat_caste(strat1, strat2, mode='collective', 
                 title=f"Macro View: Stochastic Robustness (GDP) Gen {SNAPSHOT_GEN}")
    # Force save for users with non-interactive backends
    ax_coll.figure.savefig("strat_caste_collective.png", dpi=300, bbox_inches='tight')
    print("Saved 'strat_caste_collective.png'")

    # B. The Micro View: Diversity & Merit (Individual Mode)
    ax_ind = mb.view.strat_caste(strat1, strat2, mode='individual', 
                 title=f"Micro View: Internal Diversity (Per Capita) Gen {SNAPSHOT_GEN}")
    ax_ind.figure.savefig("strat_caste_individual.png", dpi=300, bbox_inches='tight')
    print("Saved 'strat_caste_individual.png'")
    
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
