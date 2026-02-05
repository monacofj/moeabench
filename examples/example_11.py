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
    # --- CASE 1: NSGA-III (Standard Benchmark) ---
    print("\n[Case 1] NSGA-III Performance")
    exp_n3 = mb.experiment()
    exp_n3.mop = mb.mops.DTLZ2(M=3)
    exp_n3.moea = mb.moeas.NSGA3(population=92) # Standard population
    exp_n3.run(generations=150)

    # Explicitly show Data vs Reference (Optimal Front)
    mb.view.topo_shape(exp_n3, exp_n3.optimal_front(), title="Geometry: NSGA-III")
    mb.diagnostics.audit(exp_n3).report_show()
    
    # --- CASE 2: MOEA/D (Potential Distribution Bias) ---
    print("\n[Case 2] MOEA/D Performance")
    exp_md = mb.experiment()
    exp_md.mop = mb.mops.DTLZ2(M=3)
    exp_md.moea = mb.moeas.MOEAD(population=92) 
    exp_md.run(generations=150)

    # Explicitly show Data vs Reference
    mb.view.topo_shape(exp_md, exp_md.optimal_front(), title="Geometry: MOEA/D")
    mb.diagnostics.audit(exp_md).report_show()

    print("\nDiagnostics showcase completed.")

if __name__ == "__main__":
    main()
