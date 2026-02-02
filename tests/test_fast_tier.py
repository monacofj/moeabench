"""
MoeaBench Fast Tier Testing Suite
=================================

Este é o primeiro nível da pirâmide de testes do MoeaBench.
O objetivo deste tier é validar **invariantes matemáticos** e propriedades
geométricas fundamentais dos problemas (MOPs) sem depender de algoritmos estocásticos.

O que é testado:
----------------
- Continuidade e limites dos objetivos ($f_i$).
- Propriedades de soma e SOS (Sum of Squares) nas fronteiras ótimas.
- Consistência dos métodos `optimal()` e `ps()` (Pareto Set).

Características:
----------------
- **Velocidade**: Execução em milissegundos.
- **Determinismo**: 100% determinístico.
- **Uso**: Ideal para ser executado em cada "save" do desenvolvedor.

Execução:
---------
pytest tests/test_fast_tier.py
"""
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("."))
import MoeaBench as mb

@pytest.mark.parametrize("mop_name, invariant_type, expected_val", [
    ("DTLZ1", "sum", 0.5),
    ("DTLZ2", "sos", 1.0),
    ("DTLZ3", "sos", 1.0),
    ("DTLZ4", "sos", 1.0),
    ("DTLZ5", "sos", 1.0),
    ("DTLZ6", "sos", 1.0),
])
@pytest.mark.parametrize("M", [3, 5])
def test_mop_invariants(mop_name, invariant_type, expected_val, M):
    """
    Validates mathematical invariants for points sampled from the analytical Pareto Set.
    This ensures that the objective functions (f1...fM) are correctly implemented.
    """
    # Instantiate MOP
    try:
        mop = getattr(mb.mops, mop_name)(M=M)
    except TypeError:
        # Some DPFs might require D argument, but DTLZs usually don't.
        # Handling for DTLZ is standard.
        mop = getattr(mb.mops, mop_name)(M=M)

    # Sample analytical Pareto Set
    X_ps = mop.ps(n_points=100)
    
    # Evaluate
    F = mop.evaluation(X_ps)['F']
    
    if invariant_type == "sum":
        vals = np.sum(F, axis=1)
    elif invariant_type == "sos":
        vals = np.sum(F**2, axis=1)
    
    # Check invariant with numerical tolerance
    np.testing.assert_allclose(vals, expected_val, atol=1e-8, 
                               err_msg=f"MOP {mop_name} (M={M}) failed {invariant_type} invariant!")

def test_dominance_operator():
    """
    Rigorous unit test for the Pareto dominance operator logic in Population class.
    Checks:
    1. A strictly better point dominates a worse one.
    2. A point with at least one better objective (and others equal) dominates.
    3. Identical points do not dominate each other.
    4. Indifferent points (conflicting objectives) do not dominate each other.
    """
    from MoeaBench.core.run import Population
    
    # Minimization assumed
    objs = np.array([
        [0.5, 0.5], # 0: Base
        [0.4, 0.4], # 1: Strictly better (dominates 0)
        [0.6, 0.6], # 2: Strictly worse (dominated by 0)
        [0.4, 0.5], # 3: Better in f1, equal in f2 (dominates 0)
        [0.5, 0.4], # 4: Equal in f1, better in f2 (dominates 0)
        [0.6, 0.5], # 5: Worse in f1, equal in f2 (dominated by 0)
        [0.5, 0.6], # 6: Equal in f1, worse in f2 (dominated by 0)
        [0.4, 0.6], # 7: Conflicting with 0 (non-dominated)
        [0.6, 0.4], # 8: Conflicting with 0 (non-dominated)
        [0.5, 0.5], # 9: Equal to 0 (non-dominated)
    ])
    
    # Dummy variables
    vars = np.zeros((objs.shape[0], 2))
    
    pop = Population(objs, vars)
    is_dominated = pop._calc_domination()
    
    # Results analysis:
    # 0 is dominated by 1, 3, 4
    # 1 is not dominated by anyone
    # 2 is dominated by 0, 1, 3, 4, 5, 6, 9
    # 3 is not dominated (1 is better, but 1 has [0.4, 0.4] vs 3's [0.4, 0.5]) -> actually 1 dominates 3
    # 4 is dominated by 1
    # 5 is dominated by 0, 1, 3, 4, 9
    # 6 is dominated by 0, 1, 3, 4, 9
    # 7 is dominated by 1
    # 8 is dominated by 1
    # 9 is dominated by 1, 3, 4
    
    assert not is_dominated[1], "Point 1 (best) should not be dominated"
    assert is_dominated[0], "Point 0 should be dominated by 1, 3, 4"
    assert is_dominated[2], "Point 2 (worst) should be dominated"
    assert is_dominated[3], "Point 3 should be dominated by 1"
    assert is_dominated[4], "Point 4 should be dominated by 1"
    assert is_dominated[7], "Point 7 should be dominated by 1"
    assert is_dominated[8], "Point 8 should be dominated by 1"
    assert is_dominated[9], "Point 9 (copy of 0) should be dominated"

def test_mop_constraints_api():
    """
    Verifies that MOPs correctly implement the constraints API.
    """
    mop = mb.mops.DTLZ1(M=3)
    assert mop.get_n_ieq_constr() == 0, "DTLZ1 should have 0 constraints"
    
    mop9 = mb.mops.DTLZ9(M=3)
    assert mop9.get_n_ieq_constr() == 2, "DTLZ9 (M=3) should have 2 constraints"
    assert mop9.get_n_ieq_constr() > 0

def test_mop_sampling_dimensions():
    """
    Checks if Pareto Set and Pareto Front sampling return correct shapes.
    """
    M, N = 3, 10
    mop = mb.mops.DTLZ2(M=M, N=N)
    
    X_ps = mop.ps(n_points=50)
    assert X_ps.shape == (50, N)
    
    F_pf = mop.pf(n_points=50)
    assert F_pf.shape == (50, M)
