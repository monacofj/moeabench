#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 12: Finite Approximation-Induced Resolution (FAIR) Metrics
-----------------------------------------------------
For detailed documentation on FAIR metrics, see `docs/fair.md`.

This example focuses on the "Facts" layer of the diagnostic suite.
Unlike Q-Scores (which are calibrated interpretations), FAIR metrics 
provide raw, physical measurements of the population's health.

We simulate a "Premature Convergence" scenario where the algorithm 
has reached the Pareto surface but hasn't yet spread out uniformly,
leaving large gaps and irregular clusters.
"""

import mb_path
import numpy as np
import moeabench as mb

def main():
    mb.system.version()

    # Setup: We use DTLZ2 (3 objectives) as our benchmark
    mop = mb.mops.DTLZ2(M=3)
    gt = mop.pf(n_points=500)
    
    # --- SCENARIO: Good Closeness but Incomplete Coverage ---
    # We run NSGA-II for a limited duration (30 generations).
    # At this stage, it usually "touches" the front but hasn't filled the gaps.
    
    exp1 = mb.experiment()
    exp1.mop = mop
    exp1.moea = mb.moeas.NSGA2(population=100, generations=30)
    exp1.run()
    
    # 1. Visual Evidence
    mb.view.topology(exp1, gt, 
                       title="Physical Pathology: Premature Convergence",
                       labels=["Early Population", "Optimal Front (GT)"],
                       show=False) # Headless mode safety

    # 2. Individual FAIR Metrics (Manual Calculation)
    
    # A. Closeness (Physical Proximity Distribution)
    # This is the raw data used for the "Validation" layer.
    u_dist = mb.clinic.closeness(exp1, ref=gt)
    
    # B. Scalar Clinical/Fair Results
    # We compute the scalar versions of FAIR metrics
    f_cov = mb.clinic.coverage(exp1, ref=gt)
    f_gap = mb.clinic.gap(exp1, ref=gt)
    

    # 3. Consolidated FAIR Audit
    # This aggregates all FAIR metrics (Closeness, Coverage, Gap, Regularity, Balance)
    mb.clinic.audit(exp1, ground_truth=gt, quality=False).fr.report()

    # 4. Full Diagnostic Biopsy (Executive Narrative)
    diag_res = mb.clinic.audit(exp1, ground_truth=gt)


if __name__ == "__main__":
    main()
