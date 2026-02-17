#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 12: Physical Engineering Audit (FAIR Metrics)
-----------------------------------------------------
This example focuses on the "Facts" layer of the diagnostic suite.
Unlike Q-Scores (which are calibrated interpretations), FAIR metrics 
provide raw, physical measurements of the population's health.
"""

import mb_path
import numpy as np
from MoeaBench import mb

def main():
    print("Example 12: Exploring the Physical (FAIR) Layer")
    print("===============================================")

    # Setup: We use DTLZ2 (3 objectives) as our benchmark
    mop = mb.mops.DTLZ2(M=3)
    gt = mop.pf(n_points=500)
    
    # --- SCENARIO: Good Closeness but Limited coverage (Central Gap) ---
    # We simulate a "Healthy" run that has a critical structural flaw.
    print("\n[Scenario] Simulating 'Gapped Coverage': Good Convergence, but missing center.")
    
    exp = mb.experiment()
    exp.mop = mop
    exp.moea = mb.moeas.NSGA2(population=100)
    exp.run(generations=150) # Let it converge
    
    # Manually introduce a pathology: Remove the central third of the front
    # Objectives in DTLZ2 are in [0, 1]. Central regions have moderate f1, f2, f3.
    objs = exp.last_pop.objectives
    # Mask out points where f1 is between 0.3 and 0.6
    mask = (objs[:, 0] < 0.35) | (objs[:, 0] > 0.65)
    objs_gapped = objs[mask]
    exp.last_pop.objectives = objs_gapped
    
    # 1. Visual Evidence
    print("\nDisplaying Topology Shape...")
    mb.view.topo_shape(exp, gt, 
                       title="Simulated Pathology: Central Gap",
                       labels=["Gapped Population", "Optimal Front (GT)"],
                       show=False)

    # 2. Individual FAIR Metrics (Manual Calculation)
    print("\nStep 1: Calculating Individual Physical Metrics (Closeness & Coverage)...")
    
    # A. Closeness (Physical Proximity Distribution)
    # This is the raw data used for the "Certification" layer.
    u_dist = mb.diagnostics.fair_closeness(exp, ground_truth=gt)
    print("\n--- Physical Insight: Closeness (Raw Distribution) ---")
    print(f"- Mean Distance: {np.mean(u_dist):.4f} resolution-units")
    print(f"- Max Distance (95th percentile): {np.percentile(u_dist, 95):.4f}")
    
    # B. Scalar Clinical/Fair Results
    diag = mb.diagnostics.audit(exp, ground_truth=gt)
    
    if diag.fair_audit_res is None:
        print("\nWarning: Audit synthesis failed (check MOP/K calibration).")
        return

    f_res = diag.fair_audit_res.metrics
    
    print("\n--- Physical Insight: Coverage (Scalar Status) ---")
    f_res["COVERAGE"].report_show()

    # 3. Consolidated FAIR Audit
    print("\nStep 2: Performing a Full Physical Engineering Audit...")
    mb.diagnostics.fair_audit(exp, ground_truth=gt).report_show()

    print("\nExample 12 completed.")

if __name__ == "__main__":
    main()
