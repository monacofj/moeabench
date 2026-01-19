#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 10: Topological Equivalence (dist_match)
-----------------------------------------------
This example demonstrates how to use the 'dist_match' tool to determine 
if two different algorithms have converged to statistically equivalent 
distributions in both the objective and decision spaces.
"""

import mb_path
from MoeaBench import mb

def main():
    # 1. Setup: Compare NSGA-II vs NSGA-III on DTLZ2
    # We use a 3-objective problem (M=3)
    exp1 = mb.experiment()
    exp1.name = "NSGA-II"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA2(population=48, generations=50)

    exp2 = mb.experiment()
    exp2.name = "NSGA-III"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.NSGA3(population=48, generations=50)

    # 2. Execution: Run both experiments
    print("Running NSGA-II...")
    exp1.run()
    
    print("Running NSGA-III...")
    exp2.run()

    # 3. Topological Analysis: Objective Space
    # By default, dist_match(exp1, exp2) uses space='objs' (the elite front)
    print("\n--- [Analysis 1] Convergence Equivalence (Objective Space) ---")
    res_objs = mb.stats.dist_match(exp1, exp2)
    print(res_objs.report())

    # 4. Topological Analysis: Decision Space
    # We can also check if they found the same solutions in the decision space
    print("\n--- [Analysis 2] Strategy Equivalence (Decision Space) ---")
    res_vars = mb.stats.dist_match(exp1, exp2, space='vars')
    print(res_vars.report())

    # 5. Advanced: Earth Mover Distance (Geometric Distance)
    # Quantify "how far" the distributions are from each other
    print("\n--- [Analysis 3] Geometric Distance (EMD) ---")
    res_emd = mb.stats.dist_match(exp1, exp2, method='emd')
    print(f"Global Status: {res_emd.is_consistent}")
    print(f"Failed Axes (Divergent): {res_emd.failed_axes}")

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# Unlike performance metrics (Hypervolume), which tell us "who is better", 
# 'dist_match' tells us "are they finding the same thing?".
#
# 1. Objective Space Match: If 'dist_match' reports a match in objective space,
#    it means both algorithms provided a statistically similar approximation 
#    of the Pareto Front.
#
# 2. Decision Space Divergence: Sometimes, algorithms might reach the same 
#    quality (objectives) but through different sets of variables (multimodality).
#    Divergence in 'space=vars' reveals these different search strategies.
