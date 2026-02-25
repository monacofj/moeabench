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
        for K in [100]:
            if K > len(pf): continue
            
            print(f"Testing {mop.__class__.__name__} with K={K}...")
            
            # Sub-sample GT to simulate a perfect algorithm result
            idx = np.random.choice(len(pf), K, replace=False)
            perf_pop = pf[idx]
            
            # Audit directly with the objectives array and MOP reference
            result = audit(perf_pop, ground_truth=pf, problem=mop.__class__.__name__)
            
            print(f"  Status: {result.status.name}")
            print(f"  Summary: {result.summary()}")
            
            # Check Q-Scores (Should be near 1.0 for GT subset)
            for name, score in result.q_audit_res.scores.items():
                v = float(score.value)
                print(f"  - {name}: {v:.4f}")
                assert v > 0.5, f"Q-Score {name} ({v:.4f}) too low for a perfect GT subset!"
            
            # Assertions
            assert result.status == DiagnosticStatus.IDEAL_FRONT
            
    print("\nSUCCESS: Metrology is sane (Efficiency ~ 1.0 for GT sub-samples).")

if __name__ == "__main__":
    try:
        test_gt_sanity()
    except Exception as e:
        print(f"\nFAILURE: {e}")
        sys.exit(1)
