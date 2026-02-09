# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import numpy as np
import pandas as pd

# Ensure Project Root is in path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from MoeaBench.diagnostics.auditor import audit
from MoeaBench.mops import DTLZ2, DPF4
from MoeaBench.diagnostics.enums import DiagnosticStatus

def test_gt_sanity():
    print("MoeaBench Metrology Sanity Check")
    print("================================")
    
    test_cases = [DTLZ2, DPF4]
    
    for mop_cls in test_cases:
        mop = mop_cls()
        pf = mop.pf() if hasattr(mop, 'pf') else mop.optimal_front()
        
        # Test for multiple K in our grid
        for K in [100, 400]:
            if K > len(pf): continue
            
            print(f"Testing {mop.__class__.__name__} with K={K}...")
            
            # Sub-sample GT to simulate a perfect algorithm result
            idx = np.random.choice(len(pf), K, replace=False)
            perf_pop = pf[idx]
            
            # Wrap in a dummy object that audit() can handle
            class DummyRun:
                def __init__(self, objs, prob):
                    self.last_pop = type('P', (), {'objectives': objs})
                    self.experiment = type('E', (), {'mop': prob})
            
            run = DummyRun(perf_pop, mop)
            result = audit(run)
            
            print(f"  Status: {result.status.name}")
            igd_eff = result.metrics.get('igd_eff', 999)
            emd_eff = result.metrics.get('emd_eff_uniform', 999)
            
            print(f"  IGD_eff: {igd_eff:.4f}x")
            print(f"  EMD_eff: {emd_eff:.4f}x")
            
            # Assertions
            assert result.status == DiagnosticStatus.IDEAL_FRONT
            # Allow Super-Saturation (eff < 1.0) down to 0.1x
            # Upper bound 1.6x ensures we don't regress to "Generous" behavior
            assert 0.1 < igd_eff < 1.6, f"IGD efficiency {igd_eff} out of bounds for GT"
            assert 0.1 < emd_eff < 1.6, f"EMD efficiency {emd_eff} out of bounds for GT"
            
    print("\nSUCCESS: Metrology is sane (Efficiency ~ 1.0 for GT sub-samples).")

if __name__ == "__main__":
    try:
        test_gt_sanity()
    except Exception as e:
        print(f"\nFAILURE: {e}")
        sys.exit(1)
