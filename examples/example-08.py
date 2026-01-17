#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 08: Advanced Population Diagnostics Workshop
--------------------------------------------------
This example demonstrates the complete diagnostic suite of MoeaBench, 
analyzing the "Population Geology" (internal dominance structure) 
of different search algorithms. 
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt

def main():
    print("--- Advanced Diagnostics Workshop ---\n")

    # 1. Setup: Deceptive 3D problem (DTLZ3)
    mop1 = mb.mops.DTLZ3(M=3)
    repeats = 3 

    exp1 = mb.experiment()
    exp1.name = "NSGA-II"
    exp1.mop = mop1
    exp1.moea = mb.moeas.NSGA2deap(population=100, generations=52, seed=1)

    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mop1
    exp2.moea = mb.moeas.SPEA2(population=100, generations=52, seed=42)

    print(f"Running experiments ({repeats} repeats each)...")
    exp1.run(repeat=repeats)
    exp2.run(repeat=repeats)

    # 2. Analysis: Snapshot at Early Search (Gen 5)
    SNAPSHOT_GEN = 5
    print(f"\n--- Diagnostic Snapshot at Gen {SNAPSHOT_GEN} ---")
    
    # strat1 and strat2 contain:
    #             .ranks           distribution of solutions per rank
    #             .quality         metric values per rank
    #             .selection_pressure   diagnostic coefficient
    #             .report()        narrative diagnosis
    strat1 = mb.stats.strata(exp1, gen=SNAPSHOT_GEN)
    strat2 = mb.stats.strata(exp2, gen=SNAPSHOT_GEN)

    print(strat1.report())
    print("\n" + strat2.report())

    # 3. Statistical Contrast (EMD)
    # res1 contains:
    #             .value           Earth Mover's Distance
    #             .report()        narrative summary
    res1 = mb.stats.emd(strat1, strat2)
    print("\n" + res1.report())

    # 4. Visual Workshop
    print("\nGenerating visual profiles... (Close plots to finish)")
    
    mb.view.rankplot(strat1, strat2, labels=[exp1.name, exp2.name], 
                title=f"Structural Perspective (Gen {SNAPSHOT_GEN})")
    
    # B. Hierarchical View (Floating Ranks/Quality)
    mb.view.casteplot(strat1, strat2, 
                 title=f"Hierarchical Perspective: Caste Profile (Gen {SNAPSHOT_GEN})")

    # C. Competitive View (The Tier Duel)
    # New analysis: joint stratification to see cross-dominance (F1 Metaphor).
    print(f"\n--- Competitive Tier Duel (F1 Pole/Gap) ---")
    res_tier = mb.stats.tier(exp1, exp2, gen=SNAPSHOT_GEN)
    print(res_tier.report())
    mb.view.tierplot(res_tier, title="Competitive Perspective: Tier Duel")

    plt.show()

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# This analysis looks "under the hood" of the Pareto Front. 
# We compare two archetypes:
#
# 1. NSGA-II (The Phalanx): Often moves as a unified wave with many solutions 
#    clustered in a few ranks. This indicates high social cohesion in the search.
#
# 2. SPEA2 (The Sniper): Tends to promote a very sharp elite (Rank 1) while 
#    leaving the rest of the population more scattered.
#
# Scientific Perspectives (mb.view):
# - 'rankplot' (Structural): Shows the global frequency of each rank.
# - 'casteplot' (Hierarchical): Shows quality and density per rank.
# - 'tierplot' (Competitive): The Tier Duel view for direct algorithm duels.
#
# The 'casteplot' (floating ranks) proves that the population is robust: 
# early ranks are often clustered near the 1.0 quality ceiling. This means 
# that even if the absolute best solutions are lost, the "successors" are 
# functionally equivalent.
