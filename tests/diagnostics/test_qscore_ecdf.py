
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest
import numpy as np
from MoeaBench.diagnostics import qscore

class TestQScoreECDF(unittest.TestCase):
    
    def setUp(self):
        # Create a synthetic ECDF for testing
        # 200 values uniformly distributed in [0, 100]
        self.rand_ecdf = np.linspace(0, 100, 200)
        self.rand50 = 50.0 # Median
        self.ideal = 0.0
        
    def test_boundaries(self):
        # Case 1: Perfect Score (Ideal)
        q = qscore._compute_q_ecdf(self.ideal, self.ideal, self.rand50, self.rand_ecdf)
        self.assertAlmostEqual(q, 1.0, msg="Q(ideal) should be 1.0")
        
        # Case 2: Random Score
        q = qscore._compute_q_ecdf(self.rand50, self.ideal, self.rand50, self.rand_ecdf)
        self.assertAlmostEqual(q, 0.0, msg="Q(rand50) should be 0.0")
        
        # Case 3: Worse than Random
        q = qscore._compute_q_ecdf(150.0, self.ideal, self.rand50, self.rand_ecdf)
        self.assertEqual(q, 0.0, msg="Q(worse_than_random) should be 0.0 (clipped)")
        
    def test_monotonicity(self):
        # Case: Better value (closer to ideal) should have higher Q
        val_a = 10.0
        val_b = 40.0 # Worse
        
        q_a = qscore._compute_q_ecdf(val_a, self.ideal, self.rand50, self.rand_ecdf)
        q_b = qscore._compute_q_ecdf(val_b, self.ideal, self.rand50, self.rand_ecdf)
        
        self.assertGreater(q_a, q_b, f"Q({val_a}) should be > Q({val_b})")
        
    def test_degenerate_baseline(self):
        # Case: Ideal == Random (Degenerate)
        # ECDF is all zeros (e.g., perfect convergence always)
        deg_ecdf = np.zeros(200)
        deg_rand = 0.0
        deg_ideal = 0.0
        
        # Logic says: If ideal ~ rand, we return 1.0 if fair <= ideal, else 0.0
        
        # Match
        q = qscore._compute_q_ecdf(0.0, deg_ideal, deg_rand, deg_ecdf)
        self.assertEqual(q, 1.0)
        
        # Fail
        q = qscore._compute_q_ecdf(1.0, deg_ideal, deg_rand, deg_ecdf)
        self.assertEqual(q, 0.0)

    def test_interpolation(self):
        # Case: 25.0 is exactly half-way in probability space between 0 (ideal) and 50 (rand)
        # F(0) = 0.0, F(50) = 0.5. Denom = 0.5.
        # F(25) = 0.25 (roughly, due to index/N discretization)
        # Error = (0.25 - 0) / (0.5 - 0) = 0.5
        # Q = 1 - 0.5 = 0.5
        
        val = 25.0
        # Manual check of indices in linspace(0, 100, 200)
        # 0 is at index 0. 100 is at index 199.
        # 50 is at index ~100. F_rand ~ 100/200 = 0.5
        # 25 is at index ~50. F_fair ~ 50/200 = 0.25
        
        q = qscore._compute_q_ecdf(val, self.ideal, self.rand50, self.rand_ecdf)
        self.assertAlmostEqual(q, 0.5, delta=0.02) # Allow small discretization error

if __name__ == '__main__':
    unittest.main()
