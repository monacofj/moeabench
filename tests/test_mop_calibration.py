# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
import os
import sys
import shutil
import numpy as np

# Ensure project root is in path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import MoeaBench as mb
from MoeaBench.mops.base_mop import BaseMop

class DummySphere(BaseMop):
    """Simple Sphere MOP for testing calibration."""
    def __init__(self, **kwargs):
        super().__init__(name="DummySphere_v1", M=2, N=5, **kwargs)
        self.xl = np.zeros(5)
        self.xu = np.ones(5)
    
    def evaluation(self, X, **kwargs):
        # f1 = sum(x^2), f2 = sum((x-1)^2)
        f1 = np.sum(X**2, axis=1)
        f2 = np.sum((X-1)**2, axis=1)
        return {'F': np.column_stack([f1, f2])}
        
    def ps(self, n):
        # Pareto Set is x1=x2=...=x5 = t where t in [0,1]
        t = np.linspace(0, 1, n)
        return np.column_stack([t]*5)

class TestMOPCalibration(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_calibration"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.json_path = os.path.join(self.test_dir, "DummySphere_v1.json")
        if os.path.exists(self.json_path):
            os.remove(self.json_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_full_calibration_workflow(self):
        mop = DummySphere()
        
        # 1. Calibrate (Fresh)
        # Using Mop-level method
        print("\nRunning fresh calibration...")
        # Reduce k_grid for speed in tests
        recal = mop.calibrate(source_baseline=self.json_path, k_values=[20, 50])
        self.assertTrue(recal, "Should have performed recalibration")
        self.assertTrue(os.path.exists(self.json_path), "Sidecar JSON should exist")
        
        # 2. Verify File Content
        import json
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(data["problem_id"], "DummySphere_v1")
        self.assertIn("gt_reference", data)
        self.assertIn("problems", data)
        self.assertIn("20", data["problems"]["DummySphere_v1"])
        
        # 3. Reload (Idempotency)
        print("Running reload (should be instant)...")
        recal_2 = mop.calibrate(source_baseline=self.json_path)
        self.assertFalse(recal_2, "Should have loaded from cache, not recalibrated")
        
        # 4. Diagnostic Integration
        # Create a population to test Q-Score
        pop = np.random.rand(50, 2) # 50 solutions, 2 objectives
        
        # This should work now because "DummySphere_v1" is registered
        res = mb.diagnostics.q_closeness(pop, problem="DummySphere_v1", k=50)
        q = res.value
        print(f"Calculated Q-Closeness: {q}")
        self.assertGreaterEqual(q, 0.0)
        self.assertLessEqual(q, 1.0)

if __name__ == "__main__":
    unittest.main()
