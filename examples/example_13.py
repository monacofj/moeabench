#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 13: Reference-based Quality Validation (Q-Scores)
=====================================================

This example demonstrates the "Clinical Validation" layer of the diagnostic suite.
We simulate a "Collapsed Front" pathology by running a real optimizer (NSGA-II)
with 285 generations).

This forces the algorithm to fail in converging, producing a population that is:
1. "Remote" (Far from the optimal front)
2. "Collapsed" (Likely clustered in the initial random area or just beginning to move)
3. "Noise-dominant" (High entropy, low structure)

The Q-Score system will automatically detect these failures and assign "Red" verdicts.
"""

import mb_path
import moeabench as mb
import matplotlib.pyplot as plt

def main():
    mb.system.version()

    # 1. Setup: DTLZ1 (3 Objectives) and NSGA-II
    # Using the specific configuration requested by the user:
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ1(M=3)
    exp.moea = mb.moeas.NSGA2(population=100, generations=280)
    exp.name = "Strangled NSGA-II"

    # 2. Execution: Run the optimization with seed=1
    exp.run(seed=1)

    # 3. Validation Data: Ground Truth
    # We use a standard analytical optimal front (a point cloud).
    gt = exp.optimal_front()

    # 4. Run Clinical Audit (Q-Scores)
    q_res = mb.clinic.audit(exp, ground_truth=gt).quality

    # 5. Report Results
    q_res.report(full=True)
    
    # 6. Visual Confirmation
    
    # Standard call as requested: Population object vs Optimal Front points.
    
    # User requested specific block for topology:
    exp = mb.experiment()
    exp.moea = mb.moeas.NSGA2()
    exp.mop = mb.mops.DTLZ1()
    exp.population = 100
    exp.generations = 300
    exp.run(seed=1)
    mb.view.topology(exp, exp.optimal_front(), markers=True)

if __name__ == "__main__":
    main()
