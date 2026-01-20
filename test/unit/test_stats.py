# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import numpy as np

# Ensure the library is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from MoeaBench import mb

def test_perf_taxonomy():
    """Verify performance comparison functions."""
    # Synthetic data: Algorithm A significantly better than B
    data_a = np.random.normal(1.0, 0.1, 30)
    data_b = np.random.normal(0.5, 0.1, 30)
    
    # 1. perf_evidence (Mann-Whitney)
    res = mb.stats.perf_evidence(data_a, data_b)
    assert res.p_value < 0.05
    assert "favoring Group A" in res.report()
    
    # 2. perf_prob (A12 Win Probability)
    val = mb.stats.perf_prob(data_a, data_b)
    assert val.value > 0.9  # Nearly certain A > B
    
    # 3. perf_dist (KS Test)
    res_ks = mb.stats.perf_dist(data_a, data_b)
    assert res_ks.significant == True

def test_topo_attain():
    """Verify Empirical Attainment Functions logic."""
    # Create two experiments
    exp1 = mb.experiment()
    exp1.mop = mb.mops.DTLZ2(M=2)
    # NSGA-II tends to be better than a random search (SPEA2 with 1 gen)
    exp1.moea = mb.moeas.NSGA2deap(population=20, generations=20)
    exp1.run(repeat=5)
    
    # Median attainment
    surf = mb.stats.topo_attain(exp1, level=0.5)
    assert hasattr(surf, 'volume')
    assert surf.shape[1] == 2
    
    # topo_gap (Comparison)
    # Comparing exp1 with itself at different levels should show gap
    res = mb.stats.topo_gap(exp1, exp1, level=0.5)
    assert res.volume_diff == 0.0 # Same experiment
    assert "Comparison Report" in res.report()

def test_topo_dist():
    """Verify multi-axial topological matching."""
    # Fronts that match perfectly (self-comparison)
    data = np.random.random((50, 2))
    res = mb.stats.topo_dist(data, data, method='ks')
    
    assert res.is_consistent is True
    assert len(res.results) == 2 # 2 axis tested
    assert "CONSISTENT" in res.report()
    
    # Fronts that differ significantly on one axis
    data2 = data.copy()
    data2[:, 0] += 5.0 # Shift X axis
    res2 = mb.stats.topo_dist(data, data2, method='ks')
    assert res2.is_consistent is False
    assert 0 in res2.failed_axes
