#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 11: Automated Diagnostics (Algorithmic Pathology)
---------------------------------------------------------
This example demonstrates how to use the automated diagnostics module to 
interpret the health of an optimization search beyond raw numbers.
"""

import mb_path
from MoeaBench import mb

def main():
    print(f"MoeaBench v{mb.system.version()}")
    # --- CASE 1: NSGA-III on DPF1 (Optimal Solution) ---
    print("\n[Case 1] NSGA-III on DPF1 (Linear Front)")
    exp1 = mb.experiment()
    exp1.mop = mb.mops.DPF1(M=3) # Standard linear front
    exp1.moea = mb.moeas.NSGA3(population=92)
    exp1.run(generations=150)

    # Explicitly show Data vs Reference
    mb.view.topo_shape(exp1, exp1.optimal_front(), title="Geometry: NSGA-III (Healthy)")
    mb.diagnostics.audit(exp1).report_show()
    
    # --- CASE 2: MOEA/D on DPF2 (Diversity Collapse Vulnerability) ---
    print("\n[Case 2] MOEA/D on DPF2 (The Paradox of Collapse)")
    exp2 = mb.experiment()
    exp2.mop = mb.mops.DPF2(M=3) # Degenerate front known for diversity issues
    exp2.moea = mb.moeas.MOEAD(population=92) 
    exp2.run(generations=150)

    # Visualizing the pathology (Clumping should be visible)
    mb.view.topo_shape(exp2, exp2.optimal_front(), title="Geometry: MOEA/D (Collapse)")
    mb.diagnostics.audit(exp2).report_show()

    print("\nDiagnostics showcase completed.")

if __name__ == "__main__":
    main()
