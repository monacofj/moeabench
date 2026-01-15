#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 02: Algorithm Comparison (NSGA-III vs SPEA2)
---------------------------------------------------
This example illustrates how to compare two different algorithms on 
the same problem, visualizing their final fronts and convergence rates.
"""

import mb_path
from MoeaBench import mb

def main():
    # 1. Setup: Compare two algorithms on DTLZ2
    exp1 = mb.experiment()
    exp1.name = "NSGA-III"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=50, generations=50)

    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=50, generations=50)

    # 2. Execution: Run both experiments
    print(f"Running {exp1.name}...")
    exp1.run()
    
    print(f"Running {exp2.name}...")
    exp2.run()
    
    # 3. Comparative Visualization (Spatial)
    # Passing multiple experiments to spaceplot overlays their fronts.
    print("Comparing fronts...")
    mb.spaceplot(exp1, exp2, title="Comparison: NSGA-III vs SPEA2")

    # 4. Comparative Visualization (Temporal)
    # We use a shared reference to ensure a fair hypervolume comparison.
    ref = [exp1, exp2] 

    # hv1 and hv2 contain:
    #             MetricMatrix    historical hypervolume values
    hv1 = mb.hv(exp1, ref=ref)
    hv2 = mb.hv(exp2, ref=ref)
    
    print("Comparing convergence...")
    mb.timeplot(hv1, hv2, title="Hypervolume Convergence (Fair Comparison)")

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# By plotting two algorithms together, we can visually assess their trade-offs. 
# NSGA-III (Genetic Algorithm with Reference Points) usually maintains a 
# very uniform spread across the front. 
# SPEA2 (Strength Pareto EA) might show higher density in certain regions 
# depending on its internal archive and density estimation.
#
# The fair Hypervolume comparison (using a global reference point) reveals 
# which algorithm finds a 'bigger' front and how fast it reaches maturity.
