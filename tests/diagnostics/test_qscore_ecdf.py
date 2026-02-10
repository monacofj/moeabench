# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import unittest
import numpy as np

# Ensure project root in path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from MoeaBench.diagnostics import qscore, baselines

class TestQScoreECDF(unittest.TestCase):

    def test_q_score_monotonicity(self):
        """Verify that better fair values yield better Q scores."""
        # Using internal helper directly to avoid file I/O dependency
        rand_ecdf = np.linspace(0.0, 1.0, 200) # Uniform [0, 1] baseline
        ideal = 0.0
        rand50 = 0.5
        
        # Case A: Better than ideal? (Negative error) 
        # FIT ideal is 0. If value is -0.1 (impossible usually but math should work), it clipped.
        q_super = qscore._compute_q_ecdf(-0.1, ideal, rand50, rand_ecdf)
        self.assertEqual(q_super, 1.0) # Clip to 1
        
        # Case B: Standard Range
        # Boundary Checks (The most important invariants)
        self.assertEqual(qscore._compute_q_ecdf(ideal, ideal, rand50, rand_ecdf), 1.0)
        self.assertEqual(qscore._compute_q_ecdf(rand50, ideal, rand50, rand_ecdf), 0.0)
        
        # Case C: Random and Worse
        q_rand = qscore._compute_q_ecdf(0.5, ideal, rand50, rand_ecdf)
        self.assertEqual(q_rand, 0.0)
        
        q_bad = qscore._compute_q_ecdf(0.8, ideal, rand50, rand_ecdf)
        self.assertEqual(q_bad, 0.0)
        
        # Monotonicity check
        vals = np.linspace(0.0, 1.0, 50)
        qs = [qscore._compute_q_ecdf(v, ideal, rand50, rand_ecdf) for v in vals]
        # Should be non-increasing
        for i in range(len(qs)-1):
            self.assertGreaterEqual(qs[i], qs[i+1], f"Monotonicity failed at index {i}: {qs[i]} < {qs[i+1]}")

    def test_degenerate_baseline_fail_closed(self):
        """Verify that degenerate baseline (ideal == rand) raises error."""
        rand_ecdf = np.zeros(200) # All zeros (Dirac at 0)
        ideal = 0.0
        rand50 = 0.0 # Median is 0
        
        with self.assertRaises(baselines.UndefinedBaselineError):
            qscore._compute_q_ecdf(0.1, ideal, rand50, rand_ecdf)

    def test_integration_dtlz2_fit(self):
        """Verify real baseline loading and calculation for DTLZ2."""
        # DTLZ2 exists in baselines_v4.json (we acted on that assumption)
        problem = "DTLZ2" 
        k = 100
        
        # Check explicit points
        # Ideal FIT = 0.0
        q_ideal = qscore.compute_q_fit(0.0, problem, k)
        self.assertAlmostEqual(q_ideal, 1.0)
        
        # Rand FIT
        _, rand50, _ = baselines.get_baseline_ecdf(problem, k, "fit")
        q_rand = qscore.compute_q_fit(rand50, problem, k)
        self.assertAlmostEqual(q_rand, 0.0)

if __name__ == '__main__':
    unittest.main()
