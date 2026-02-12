
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
import numpy as np
import os
import json
from MoeaBench.diagnostics import baselines

class TestGoldenDTLZ6(unittest.TestCase):
    """
    Deterministically verifies the s_K integrity for DTLZ6 (The 'Milky Way' Fix).
    Ensures that baselines v3 and the runtime logic are perfectly aligned.
    """

    def test_sk_calculation_integrity(self):
        """Verifies that get_resolution_factor_k produces the exact approved float."""
        # Mock GT for DTLZ6/200? No, we need to load the real one or rely on the function loading it.
        # baselines.get_resolution_factor_k loads U_K internally from the package.
        # We just need to call it.
        
        # We need the ground truth array to call the function.
        # The function signature is get_resolution_factor_k(gt, k, seed).
        # So we must load the DTLZ6 GT first.
        
        # Where is DTLZ6 GT?
        # It's in diagnostics/resources/references/DTLZ6/calibration_package.npz['gt_norm']
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "MoeaBench"))
        pkg_path = os.path.join(repo_root, "diagnostics/resources/references/DTLZ6/calibration_package.npz")
        
        if not os.path.exists(pkg_path):
             self.skipTest(f"DTLZ6 package not found at {pkg_path}")
             
        pkg = np.load(pkg_path)
        gt = pkg['gt_norm']
        
        # Execute Runtime Logic
        s_k = baselines.get_resolution_factor_k(gt, 200, seed=0)
        
        # Golden Value from Plan/Review
        EXPECTED_S_K = 0.00681372890931866
        
        print(f"\n[Golden Check] Runtime s_K (DTLZ6, K=200): {s_k}")
        self.assertAlmostEqual(s_k, EXPECTED_S_K, places=15, 
                               msg="Critical: s_K runtime calculation diverged from Golden Value.")

    def test_baseline_v3_migration(self):
        """Verifies that baselines_v3.json contains the correctly rescaled fit.rand50."""
        # Load v3 JSON directly
        v3_path = os.path.join(os.path.dirname(__file__), "MoeaBench/diagnostics/resources/baselines_v3.json")
        
        with open(v3_path, 'r') as f:
            data = json.load(f)
            
        # Traverse to DTLZ6 -> 200 -> fit -> rand50
        try:
            rand50 = data["problems"]["DTLZ6"]["200"]["fit"]["rand50"]
        except KeyError:
            self.fail("DTLZ6/200/fit/rand50 missing from baselines_v3.json")
            
        # Golden Value from Plan (s_gt/s_k * 209 = ~23.8)
        # Expected: 23.827020470485664
        EXPECTED_RAND = 23.827020470485664
        
        print(f"[Golden Check] V3 fit.rand50 (DTLZ6, K=200): {rand50}")
        # Allow small float wiggle room due to arch differences, but should be tiny
        self.assertAlmostEqual(rand50, EXPECTED_RAND, places=8,
                               msg="Critical: V3 Baseline migration failed to match Golden Value.")

if __name__ == '__main__':
    unittest.main()
