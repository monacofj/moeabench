# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import os
import sys

# Ensure MoeaBench is in path
sys.path.append(os.getcwd())

from MoeaBench.diagnostics.auditor import audit
from MoeaBench.diagnostics.enums import DiagnosticProfile

def debug_2min():
    print("=== MoeaBench 2-Minute Diagnostic Audit ===")
    
    # Use a problem where we saw "suspicious perfection"
    mop_name = "DTLZ2"
    from pymoo.problems import get_problem
    prob = get_problem(mop_name.lower(), n_var=10, n_obj=3)
    
    # 1. Create a "Dirty" Front (Shifted and Sparse)
    # We take the Ground Truth and add a significant bias
    pf = prob.pareto_front()
    if pf is None: pf = prob.pf()
    dirty_pop = pf + 0.1  # Shifted by 10%
    
    print(f"\nTarget: {mop_name} (Shifted by 0.1)")
    
    # 2. Run Audit
    res = audit(dirty_pop, ground_truth=pf, profile=DiagnosticProfile.RESEARCH)
    
    # 3. Print Surgical Values
    m = res.metrics
    print("\n[Internal Metrics]")
    print(f"GD_p95:     {m.get('gd_p95', 'N/A'):.6f}")
    print(f"IGD_raw:    {m.get('igd', 'N/A'):.6f}")
    print(f"EMD_raw:    {m.get('emd', 'N/A'):.6f}")
    
    print("\n[Efficiency & Normalization]")
    print(f"IGD_eff:    {m.get('igd_eff', 'N/A')}")
    print(f"EMD_eff:    {m.get('emd_eff_uniform', 'N/A')}")
    print(f"K (Snapped): {m.get('K', 'N/A')}")
    print(f"Diameter:   {res.details.get('diameter', 'N/A') if res.details else 'N/A'}")
    
    print("\n[Decision Logic]")
    print(f"H_rel:      {m.get('h_rel', 'N/A')}")
    print(f"Status:     {res.status.name}")
    print(f"Verdict:    {res.verdict}")
    print(f"Rationale:  {res.rationale()}")
    
    # 4. Check for Identical Results Bug (Simulate two different "bad" pops)
    dirty_pop_2 = pf + 0.2
    res2 = audit(dirty_pop_2, ground_truth=pf, profile=DiagnosticProfile.RESEARCH)
    print("\n[Identical Results Test]")
    print(f"Pop 1 (0.1 shift) IGD_raw: {m.get('igd'):.6f}")
    print(f"Pop 2 (0.2 shift) IGD_raw: {res2.metrics.get('igd'):.6f}")
    
    if np.isclose(m.get('igd'), res2.metrics.get('igd')):
        print("\nCRITICAL: Identical results detected for different inputs! Data shadowing confirmed.")
    else:
        print("\nSUCCESS: Auditor distinguishes between different populations.")

if __name__ == "__main__":
    debug_2min()
