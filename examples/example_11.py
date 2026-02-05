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
    print(f"MoeaBench Version: {mb.system.version()}")
    print("-" * 30)

    # 1. Setup Study: Standard Case
    print("\n[Scenario 1] Standard Optimization (NSGA-II on DTLZ2)")
    exp = mb.experiment()
    exp.name = "NSGA-II Normal"
    exp.mop = mb.mops.DTLZ2(M=3)
    exp.moea = mb.moeas.NSGA3(population=100) # Using NSGA3 for robustness
    
    # Execute
    exp.run(generations=150)

    # --- METHOD A: Manual Audit (Recommended) ---
    # We explicitly audit the experiment to see the scientific verdict.
    print("\nPerforming Manual Audit...")
    diagnosis = mb.diagnostics.audit(exp)
    
    # Display the report (In terminal it prints; in Jupyter it renders Markdown)
    diagnosis.report_show()
    
    # 2. Setup Study: Potential Diversity Collapse
    print("\n[Scenario 2] High Selection Pressure / Sparse Population")
    exp_sparse = mb.experiment()
    exp_sparse.name = "NSGA-II Sparse"
    exp_sparse.mop = mb.mops.DTLZ2(M=3)
    # Using very small population to trigger Sparse Approximation or Collapse
    exp_sparse.moea = mb.moeas.NSGA3(population=12) 
    
    # --- METHOD B: Integrated Diagnosis ---
    # We enable diagnostics directly in the run loop (caution: slower)
    print("\nRunning with Integrated Diagnostics...")
    exp_sparse.run(generations=50, diagnose=True)

    # 3. Accessing the Rationale Programmatically
    # The 'rationale' method provides the textbook-style explanation.
    print("\nInvestigating Sparse Rationale:")
    diag_sparse = mb.diagnostics.audit(exp_sparse)
    print(f"Scientific Verdict: {diag_sparse.rationale()}")

    print("\nDiagnostics showcase completed.")

if __name__ == "__main__":
    main()
