#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 12: Physical Performance Assessment (FAIR Metrics)
-----------------------------------------------------
This example focuses on the "Facts" layer of the diagnostic suite.
Unlike Q-Scores (which are calibrated interpretations), FAIR metrics 
provide raw, physical measurements of the population's health.

We simulate a "Premature Convergence" scenario where the algorithm 
has reached the Pareto surface but hasn't yet spread out uniformly,
leaving large gaps and irregular clusters.
"""

import mb_path
import numpy as np
from MoeaBench import mb

def main():
    print(f"MoeaBench v{mb.system.version()}")
    print("Example 12: Physical Performance Assessment (FAIR Metrics)")
    print("===============================================")

    # Setup: We use DTLZ2 (3 objectives) as our benchmark
    mop = mb.mops.DTLZ2(M=3)
    gt = mop.pf(n_points=500)
    
    # --- SCENARIO: Good Closeness but Incomplete Coverage ---
    # We run NSGA-II for a limited duration (30 generations).
    # At this stage, it usually "touches" the front but hasn't filled the gaps.
    print("\n[Scenario] Simulating 'Premature Convergence': On-target but clustered.")
    
    exp1 = mb.experiment()
    exp1.mop = mop
    exp1.moea = mb.moeas.NSGA2(population=100, generations=30)
    exp1.run()
    
    # 1. Visual Evidence
    print("\nDisplaying Topology Shape...")
    mb.view.topo_shape(exp1, gt, 
                       title="Physical Pathology: Premature Convergence",
                       labels=["Early Population", "Optimal Front (GT)"],
                       show=False) # Headless mode safety

    # 2. Individual FAIR Metrics (Manual Calculation)
    print("\nStep 1: Calculating Individual Physical Metrics (Closeness & Coverage)...")
    
    # A. Closeness (Physical Proximity Distribution)
    # This is the raw data used for the "Validation" layer.
    u_dist = mb.diagnostics.closeness(exp1, ground_truth=gt)
    print("\n--- Physical Insight: Closeness (Raw Distribution) ---")
    print(f"- Mean Distance: {np.mean(u_dist.raw_data):.4f} resolution-units")
    print(f"- Max Distance (95th percentile): {np.percentile(u_dist.raw_data, 95):.4f}")
    
    # B. Scalar Clinical/Fair Results
    # We compute the scalar versions of FAIR metrics
    f_cov = mb.diagnostics.coverage(exp1, ground_truth=gt)
    f_gap = mb.diagnostics.gap(exp1, ground_truth=gt)
    
    print("\n--- Physical Insight: Coverage & Gaps ---")
    print(f"- Coverage Score: {float(f_cov):.4f} (Avg distance to manifold)")
    print(f"- Max Gap Detected: {float(f_gap):.4f} (Largest hole size)")

    # 3. Consolidated FAIR Audit
    print("\nStep 2: Performing a Full Physical Engineering Audit...")
    # This aggregates all FAIR metrics (Closeness, Coverage, Gap, Regularity, Balance)
    mb.diagnostics.fair_audit(exp1, ground_truth=gt).report_show()

    # 4. Full Diagnostic Biopsy (Executive Narrative)
    print("\nStep 3: Performing Full Diagnostic Biopsy...")
    diag_res = mb.diagnostics.audit(exp1, ground_truth=gt)
    print("\n--- Executive Summary ---")
    print(diag_res.summary())

    print("\nExample 12 completed.")

if __name__ == "__main__":
    main()
