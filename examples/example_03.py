#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 03: Inspected Fronts and Convergence States
--------------------------------------------------
This example demonstrates how to extract and visualize subsets of a 
population, such as the dominated solutions vs. the Pareto Front, 
and compare early vs. late stages of a single execution.
"""

import mb_path
from MoeaBench import mb

def main():
    print(f"MoeaBench v{mb.system.version()}")
    # 1. Setup: Run a longer experiment to see convergence
    exp1 = mb.experiment()
    exp1.name = "NSGA-III"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=100, generations=100)

    # 2. Execution
    print("Running experiment...")
    exp1.run()
    
    # 3. Visualization: Front vs. Background Population (Topographic Domain)
    # We can distinguish the "winners" (Rank 1) from the rest of the search.
    print("Plotting winners vs background...")
    mb.view.topo_shape(exp1.front(), exp1.non_front(), 
                      labels=["Pareto Front", "Dominated"], 
                      title="Anatomy of a Population")

    # 4. Visualization: Search Trajectory (Topographic Domain)
    # Compare an early snapshot (Gen 5) with the final state.
    print("Comparing snapshots...")
    mb.view.topo_shape(exp1.front(5), exp1, 
                      labels=["Elite at Gen 5", "Final Elite"], 
                      title="Convergence Path")

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# Visualizing the 'non_front' (dominated solutions) reveals where the algorithm 
# has been searching. It shows the "cloud" of candidate solutions from which 
# the Pareto Front was exported.
#
# The comparison between 'front(5)' and the final front is a powerful 
# diagnostic. It shows how much 'work' the algorithm did after the initial 
# discovery phase, illustrating the refinement process toward the global optimum.
