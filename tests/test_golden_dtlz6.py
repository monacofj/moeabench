
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
import numpy as np
import os
import json
from moeabench.diagnostics import baselines


GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dtlz6.json")


class TestGoldenDTLZ6(unittest.TestCase):
    """
    Deterministically verifies the s_K integrity for DTLZ6 (The 'Milky Way' Fix).
    Ensures that baselines v3 and the runtime logic are perfectly aligned.
    """

    def test_sk_calculation_integrity(self):
        """Verifies that get_resolution_factor_k matches the frozen DTLZ6 baseline."""
        # Mock GT for DTLZ6/200? No, we need to load the real one or rely on the function loading it.
        # baselines.get_resolution_factor_k loads U_K internally from the package.
        # We just need to call it.
        
        # We need the ground truth array to call the function.
        # The function signature is get_resolution_factor_k(gt, k, seed).
        # So we must load the DTLZ6 GT first.
        
        # Where is DTLZ6 GT?
        # It's in diagnostics/resources/references/DTLZ6/calibration_package.npz['gt_norm']
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        pkg_path = os.path.join(repo_root, "moeabench", "diagnostics", "resources", "references", "DTLZ6", "calibration_package.npz")
        
        if not os.path.exists(pkg_path):
             self.skipTest(f"DTLZ6 package not found at {pkg_path}")
             
        pkg = np.load(pkg_path)
        gt = pkg['gt_norm']
        with open(GOLDEN_PATH, 'r') as f:
            golden = json.load(f)
        
        # Execute Runtime Logic
        s_k = baselines.get_resolution_factor_k(gt, 200, seed=0)
        expected_s_k = golden["DTLZ6"]["200"]["s_k_seed0"]
        
        print(f"\n[Golden Check] Runtime s_K (DTLZ6, K=200): {s_k}")
        self.assertAlmostEqual(s_k, expected_s_k, places=15,
                               msg="Critical: s_K runtime calculation diverged from Golden Value.")

    def test_baseline_v0132_integrity(self):
        """Verifies that baselines_v0.13.2.json still matches the frozen DTLZ6 check."""
        # Load JSON directly
        v_path = os.path.join(os.path.dirname(__file__), "../moeabench/diagnostics/resources/baselines_v0.13.2.json")
        with open(GOLDEN_PATH, 'r') as f:
            golden = json.load(f)
        
        with open(v_path, 'r') as f:
            data = json.load(f)
            
        # Traverse to DTLZ6 -> 200 -> closeness -> rand50
        try:
            rand50 = data["problems"]["DTLZ6"]["200"]["closeness"]["rand50"]
        except KeyError:
            self.fail("DTLZ6/200/closeness/rand50 missing from baselines_v0.13.2.json")
        
        expected_rand = golden["DTLZ6"]["200"]["closeness_rand50_v0_13_2"]
        
        print(f"[Golden Check] V0.13.2 closeness.rand50 (DTLZ6, K=200): {rand50}")
        # Allow small float wiggle room due to arch differences, but should be tiny
        self.assertAlmostEqual(rand50, expected_rand, places=8,
                               msg="Critical: Baseline integrity check failed to match Expected Value.")

if __name__ == '__main__':
    unittest.main()
