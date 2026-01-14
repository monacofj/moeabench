#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 08: Population Stratification (Rank Distribution) Analysis.

This example demonstrates how to use the 'stratification' API to analyze 
the internal dominance structure of a population and compare the selection 
pressure of two different algorithms.
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt

# 1. Setup a challenging, multi-modal benchmark (DTLZ3 with 3 objectives)
# DTLZ3 is highly deceptive; algorithms will struggle to find even 
# local fronts early on, leading to deep stratification.
mop = mb.DTLZ3(M=3)

# 2. Configure algorithms
# We'll use different seeds to ensure distinct search trajectories.
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

print(f"\n[1/2] Running {exp_nsga2.name}...")
exp_nsga2.run(repeat=3)

print(f"\n[2/2] Running {exp_spea2.name}...")
exp_spea2.run(repeat=3)

# 3. Perform Population Stratification at a snapshot in EARLY search (Gen 5)
# This is when the "internal effort" of the search is most visible.
SNAPSHOT_GEN = 5 
print(f"\n--- Diagnostic Analysis (at Gen {SNAPSHOT_GEN}) ---")

strat_nsga2 = mb.stratification(exp_nsga2, gen=SNAPSHOT_GEN)
strat_spea2 = mb.stratification(exp_spea2, gen=SNAPSHOT_GEN)

# 4. Compare Selection Pressure
# High value = aggressive (most in Rank 1/2)
# Low value = relaxed (scattered across many ranks)
print(f"NSGA-II Selection Pressure: {strat_nsga2.selection_pressure():.4f}")
print(f"SPEA2 Selection Pressure:   {strat_spea2.selection_pressure():.4f}")

# 5. Calculate Structural Difference (Earth Mover's Distance)
# Tells us how fundamentally different their population architectures are.
dist = mb.emd(strat_nsga2, strat_spea2)
print(f"Structural Difference (EMD): {dist:.4f}")

# 6. Visualize the Rank Distributions side-by-side
# Grouped bar charts allow for direct comparison of dominance depth.
print("Plotting distributions... (Close the plot to finish)")
mb.stratification_plot(strat_nsga2, strat_spea2, title=f"Dominance Depth at Gen {SNAPSHOT_GEN}")
plt.show()

# --- Interpretation of the Results ---
#
# Comparing the "Dominance Depth" reveals the distinct search "DNA" of each algorithm:
#
# 1. SPEA2: The "Elite-Driven" Specialist
#    - Result: More solutions in Rank 1, but spread across many deeper ranks (e.g., 6+).
#    - Meaning: SPEA2 acts as a "Sniper." Its selection mechanism is highly effective 
#      at promoting absolute elites to the Pareto Front, but it is "tolerant" of 
#      lower-quality solutions, allowing a long tail to persist.
#    - Verdict: Yields a sharper front but a fragmented, unequal population.
#
# 2. NSGA-II: The "Collective Wave" Generalist
#    - Result: Fewer solutions in Rank 1, but the entire population is compressed 
#      into very few ranks (e.g., only 3).
#    - Meaning: NSGA-II acts as a "Phalanx." Its tournament selection and crowding 
#      mechanism enforce a strong "Quality Discipline." It pulls the worst solutions 
#      up aggressively, moving the population as a dense, unified wave.
#    - Verdict: Yields a more robust, unified population, though it may take 
#      longer to reach the absolute peaks.
#
# 3. Earth Mover's Distance (EMD)
#    - This measures the mathematical "work" required to transform one 
#      stratification profile into the other. It quantifies how fundamentally 
#      different their search behaviors are.
