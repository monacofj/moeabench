#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 09: Polar Dominance Phase Analysis (Search DNA).

This example demonstrates the advanced 'Polar Phase' diagnostics to 
uncover the structural health and maturity of an algorithm's population.
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt

# 1. Setup a challenging, multi-modal benchmark (DTLZ3 with 3 objectives)
mop = mb.DTLZ3(M=3)

# 2. Configure algorithms
nsga2 = mb.NSGA2(population=100, generations=50, seed=1)
spea2 = mb.SPEA2(population=100, generations=50, seed=42)

exp_nsga2 = mb.experiment()
exp_nsga2.moea = nsga2
exp_nsga2.mop = mop
exp_nsga2.name = "NSGA-II"

exp_spea2 = mb.experiment()
exp_spea2.moea = spea2
exp_spea2.mop = mop
exp_spea2.name = "SPEA2"

print(f"\nRunning experiments...")
exp_nsga2.run(repeat=3)
exp_spea2.run(repeat=3)

# 3. Perform Population Stratification (Snapshot at Gen 5)
SNAPSHOT_GEN = 5 
strat_nsga2 = mb.stratification(exp_nsga2, gen=SNAPSHOT_GEN)
strat_spea2 = mb.stratification(exp_spea2, gen=SNAPSHOT_GEN)

# 4. Phase Space Metrics (GDI: Deficiency | PMI: Maturity)
print(f"\n--- Polar Phase Analysis (at Gen {SNAPSHOT_GEN}) ---")
print(f"GDI (Global Deficiency Index): Measures total search cost (Magnitude Rho)")
print(f"PMI (Population Maturity Index): Measures efficiency/landing (Angle Theta)")
print("-" * 50)
print(f"{exp_nsga2.name:>7}: GDI={strat_nsga2.gdi:.2f}, PMI={strat_nsga2.pmi:.2f}")
print(f"{exp_spea2.name:>7}: GDI={strat_spea2.gdi:.2f}, PMI={strat_spea2.pmi:.2f}")

# 5. Visualize the Polar Phase Fan
# This plot maps Rank vs Quality into a vector fan from the origin.
print("\nPlotting Phase Fan... (Close plot to finish)")
mb.polarplot(strat_nsga2, strat_spea2, title=f"Dominance Phase Fan (Gen {SNAPSHOT_GEN})")

plt.show()

# --- Interpretation of the Polar Fan ---
#
# 1. GDI (Magnitude Rho): 
#    A long vector means a solution is "heavy" with failure (high rank or low quality).
#
# 2. PMI (Angle Theta):
#    - Small angle (Horizontal) = "Landed". High quality even in deep layers.
#    - Large angle (Vertical)   = "Floating". Immature population with poor depth.
#
# 3. Algorithm Archetypes:
#    - NSGA-II: The "Phalanx". Usually shows a narrow, horizontal fan (Mature).
#    - SPEA2:   The "Sniper". Usually shows a wide, vertical fan (Elite-heavy).
