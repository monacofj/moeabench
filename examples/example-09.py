#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 09: Rank-Quality Profile Analysis.

This example demonstrates how to visualize the objective quality gradient 
across population ranks, revealing the internal anatomy of the search.
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

# 4. Phase Space Metrics
print(f"\n--- Rank-Quality Analysis (at Gen {SNAPSHOT_GEN}) ---")
print("-" * 50)
print(f"Analyzing {exp_nsga2.name} and {exp_spea2.name} profiles...")

# 5. Floating Rank Quality Profile
# X = Rank, Y = Quality (Center)
# Bar Height = Solution Density
print("\nPlotting Rank Profile... (Close plot to finish)")
mb.rankplot(strat_nsga2, strat_spea2, title=f"Floating Rank Quality (Gen {SNAPSHOT_GEN})")

plt.show()

# --- Interpretation of the Results ---
#
# 1. Floating Rank Plot (mb.rankplot):
#    - Position (Center): Shows the Quality of the rank (Default: Hypervolume).
#    - Height (Tall/Short): Shows how many solutions are "working" at that level.
#
# 2. Algorithm Archetypes:
#    - Collective Generalist (e.g. NSGA-II): Usually a few TALL bars at HIGH quality.
#    - Elite Specialist (e.g. SPEA2): Usually a SHORT bar at Rank 1 (Top Elite), 
#      followed by a trail of lower, scattered bars.
#
# 3. Diagnostic Interpretation: Layer Proximity and Search Maturity
#
# A frequent observation in the Floating Rank Profile is the presence of multiple 
# early ranks (Rank 1, 2, 3...) clustered near the 1.0 Relative Efficiency ceiling.
# This phenomenon reveals a key aspect of the "Population Geology":
#
# A. Spatial Overlap (The 'Onion' Effect):
# In a maturing search, the dominance layers are not distant islands; they are 
# thin, tightly packed "sheets" in objective space. Because Hypervolume measures 
# the region bounded by a reference point, the Rank 2 layer—which sits mere 
# fractions of a unit behind Rank 1—covers nearly the same objective territory. 
# Mathematically, the "Utility Gap" between these first layers is minimal.
#
# B. High Selection Pressure:
# When the first several ranks act as high-quality "platforms," it indicates 
# that the algorithm has successfully compressed the population towards the front. 
# There is a high concentration of competitive solutions, meaning that even 
# if the absolute "Best" (Rank 1) were lost, the "Successor" (Rank 2) is 
# functionally equivalent in terms of search coverage.
#
# C. Quality Degradation:
# The true "Diagnostic Signal" is found where the quality begins to drop. 
# A sudden cliff in HV values reveals the boundary between the active search 
# frontier and the "stale" or "exploratory" individuals of the population.
#
# In summary, the proximity of early ranks to 1.0 is not a redundancy of data, 
# but a visual proof of search maturity and structural robustness.
