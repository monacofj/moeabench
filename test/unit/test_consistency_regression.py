# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import pytest
import numpy as np

# Ensure the library is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from MoeaBench import mb

def test_dpf_dimension_validation():
    """Reproduce error_01.py: M > N should fail gracefully with ValueError."""
    # M=10, K=5, d=2 => N = 2+5-1 = 6. M > N should raise ValueError.
    with pytest.raises(ValueError) as excinfo:
        mop = mb.mops.DPF5(M=10, K=5, d=2)
    
    assert "cannot be greater than dimensionality" in str(excinfo.value)
    print("\n[PASS] DPF5 dimension validation caught invalid M > N.")

def test_dpf_initialization_success():
    """Verify that DPF problems initialize correctly with valid parameters."""
    try:
        mop = mb.mops.DPF1(M=3, D=2)
        assert mop.M == 3
        assert mop.D == 2
        assert mop._chaos_pool is not None
    except NameError as e:
        pytest.fail(f"DPF1 failed with NameError: {e}")
    print("[PASS] DPF1 initialized successfully.")

def test_custom_mop_signature():
    """Reproduce error_02.py: Custom MOP should handle **kwargs to be flexible."""
    class MyProblem(mb.mops.BaseMop):
        def __init__(self, **kwargs):
            # Safe way to default M and N while allowing overrides from kwargs
            kwargs.setdefault('M', 2)
            kwargs.setdefault('N', 10)
            super().__init__(**kwargs)

        def evaluation(self, X, n_ieq_constr=0):
            return {'F': np.zeros((X.shape[0], self.M))}

    # This should NOT raise TypeError if the user follows the **kwargs pattern
    try:
        mop = MyProblem(M=2, N=10)
        assert mop.M == 2
        assert mop.N == 10
    except TypeError as e:
        pytest.fail(f"Custom MOP failed with TypeError (did you forget **kwargs?): {e}")
    
    print("[PASS] Custom MOP signature validation passed.")

def test_deap_population_guard():
    """Verify that NSGA2deap warns/fails if population is not divisible by 4."""
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ2(M=2)
    # 10 is not divisible by 4
    exp.moea = mb.moeas.NSGA2deap(population=10, generations=2)
    
    with pytest.raises(ValueError) as excinfo:
        exp.run()
    
    assert "multiple of 4" in str(excinfo.value)
    print("[PASS] NSGA2deap population guard caught invalid size.")

def test_dtlz6_numerical_range():
    """Check DTLZ6 optimal front range."""
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ6(M=3)
    opt = exp.optimal(n_points=100)
    
    # Radius should be 1.0
    norms = np.sqrt(np.sum(opt.objectives**2, axis=1))
    assert np.allclose(norms, 1.0, atol=1e-5)
    
    # All objectives should be in [0, 1]
    assert np.all(opt.objectives >= 0.0)
    assert np.all(opt.objectives <= 1.0)
    print("[PASS] DTLZ6 numerical range is healthy.")
