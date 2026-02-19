#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Example 13: Clinical Quality Audit (Q-Scores)
=====================================================

This example demonstrates the "Clinical Validation" layer of the diagnostic suite.
We simulate a "Collapsed Front" pathology by running a real optimizer (NSGA-II)
with extremely limited resources (5 generations).

This forces the algorithm to fail in converging, producing a population that is:
1. "Remote" (Far from the optimal front)
2. "Collapsed" (Likely clustered in the initial random area or just beginning to move)
3. "Noise-dominant" (High entropy, low structure)

The Q-Score system will automatically detect these failures and assign "Red" verdicts.
"""

import mb_path
from MoeaBench import mb
import matplotlib.pyplot as plt

def main():
    print(f"MoeaBench v{mb.system.version()}")
    print("=== Example 13: Clinical Quality Audit (Q-Scores) ===\n")

    # 1. Setup: DTLZ2 (3 Objectives) and NSGA-II
    # We use a standard setup but strangle the resources to force a failure.
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ2(M=3)
    
    # "Strangled" Configuration: Only 52 individuals and 40 generations.
    # A healthy run usually requires 100+ generations.
    # Note: Population must be a multiple of 4 for minimal compatibility with TournamentDCD
    exp.moea = mb.moeas.NSGA2(population=52, generations=40)
    exp.name = "Strangled NSGA-II"

    # 2. Execution: Run the optimization
    print("Running optimization with limited resources (simulating failure)...")
    exp.run()

    # 3. Validation Data: Ground Truth
    # Needed for the audit to know what "Perfection" looks like.
    gt = exp.mop.pf(n_points=1000)

    # 4. Run Clinical Audit (Q-Scores)
    print("Running Clinical Quality Audit...")
    q_res = mb.diagnostics.q_audit(exp, ground_truth=gt)

    # 5. Report Results
    # This uses the new elegant terminal format.
    print("\n--- Clinical Quality Report ---\n")
    q_res.report_show()
    
    # 5.1 Clinical Narrative Summary (Hierarchical Decision Tree)
    print("\n--- Clinical Narrative Summary ---")
    print(q_res.summary())
    
    # 6. Visual Confirmation
    # Visually compare the "Strangled" population against the true front.
    print("\nDisplaying Topology Shape (Close setting window to finish)...")
    mb.view.topo_shape(exp, gt, 
                       title="Pathology: Resource Starvation (Collapsed Front)",
                       labels=["Strangled Pop", "Optimal Front (GT)"],
                       show=False) # Headless mode safety
    print("\nDone.")

if __name__ == "__main__":
    main()
