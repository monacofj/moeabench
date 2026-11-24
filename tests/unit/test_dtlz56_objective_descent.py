# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent tests; preserve every existing regression baseline."""
from math import cos, pi, sin

import numpy as np
import pytest

from moeabench.mops._dtlz56_objective_descent import (
    audit_objective_local_descent,
)


def _original_objective(m, g, u):
    """Published angle equations, independently from production MOP classes."""
    assert len(u) == m-1
    theta = [pi*u[0]/2]
    theta.extend(pi*(1+2*g*x)/(4*(1+g)) for x in u[1:])
    f = np.ones(m)
    for j in range(m):
        for i in range(m-j-1):
            f[j] *= cos(theta[i])
        if j > 0:
            f[j] *= sin(theta[m-j-1])
    return (1+g)*f


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [4, 5, 10])
@pytest.mark.parametrize("first", [0.2, 0.5, 0.8])
def test_objective_only_reconstruction_finds_strict_witness(problem,m,first):
    # All inner position variables are 1/2, hence a feasible same-ray
    # lower-g dominator exists. This exercises reconstruction of M>=4.
    g = 0.3
    u = [first]+[0.5]*(m-2)
    y = _original_objective(m,g,u)
    result = audit_objective_local_descent(
        y,problem=problem,k=4,tolerance=1e-9
    )
    assert result.status == "witness", result.to_dict()
    assert result.reconstruction_error < 1e-9
    x = np.asarray(result.reconstructed_decision)
    assert x.shape == (m+3,)
    assert np.all((x >= 0) & (x <= 1))
    z = np.asarray(result.local_witness["witness_objectives"])
    assert np.all(z < y-1e-9)
    assert np.linalg.norm(z) < np.linalg.norm(y)-1e-9


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_g_zero_cannot_be_declared_dominated(problem):
    y = _original_objective(4,0.0,[0.4,0.5,0.5])
    result = audit_objective_local_descent(y,problem=problem,k=4)
    assert result.status == "unverified"
    assert result.local_witness is None


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_original_evaluator_and_sampler_are_not_used(monkeypatch,problem):
    import moeabench as mb
    klass = getattr(mb.mops,problem)

    def forbidden(*args,**kwargs):
        raise AssertionError("Production evaluation/PS/PF must not be called")

    monkeypatch.setattr(klass,"evaluation",forbidden)
    monkeypatch.setattr(klass,"ps",forbidden)
    monkeypatch.setattr(klass,"pf",forbidden)
    y = _original_objective(4,0.3,[0.3,0.5,0.5])
    assert audit_objective_local_descent(
        y,problem=problem,k=4
    ).status == "witness"


@pytest.mark.parametrize("bad",[
    np.array([1.0,float("nan")]),
    np.array([-0.1,1.0]),
    np.array([4.0,4.0,4.0,4.0]),
])
def test_invalid_reference_rejected(bad):
    with pytest.raises(ValueError):
        audit_objective_local_descent(bad,problem="DTLZ5",k=4)


def test_invalid_options_rejected():
    y = _original_objective(4,0.3,[0.3,0.5,0.5])
    with pytest.raises(ValueError):
        audit_objective_local_descent(y,problem="DTLZ5",k=0)
    with pytest.raises(ValueError):
        audit_objective_local_descent(y,problem="DTLZ5",k=4,tolerance=0)
