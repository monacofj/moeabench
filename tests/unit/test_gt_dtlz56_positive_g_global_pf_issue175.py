# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New exact positive-g global-PF theorem tests (no old regression edits)."""
from copy import deepcopy
from fractions import Fraction as Q
from math import cos, sin, pi

import pytest

from scripts.audit.dtlz56_positive_g_global_pf import (
    certify_positive_g_global_pf, replay_positive_g_global_pf,
)


def _independent_original_objective(positions, g):
    """Separate published-angle implementation; no production MOP oracle."""
    m = len(positions)+1
    angle = [pi*float(positions[0])/2]
    angle.extend(pi*(1+2*float(g)*float(v))/(4*(1+float(g)))
                 for v in positions[1:])
    result = []
    for j in range(m):
        value = 1+float(g)
        for i in range(m-j-1):
            value *= cos(angle[i])
        if j:
            value *= sin(angle[m-j-1])
        result.append(value)
    return result


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [4,5,10])
@pytest.mark.parametrize("g", [Q(0),Q(1,8),Q(1,2),Q(1)])
def test_positive_g_front_stratum_has_symbolic_certificate(problem,m,g):
    positions = [Q(0),Q(1),Q(0)]+[Q(1,3)]*(m-4)
    result = certify_positive_g_global_pf(
        positions,g=g,problem=problem,k=4
    )
    assert result.conclusion == "rigorously_global_pf_member"
    assert result.to_dict()["scientific_certification"] == "unverified"
    assert replay_positive_g_global_pf(
        positions,result.to_dict(),g=g,problem=problem,k=4
    )
    y = _independent_original_objective(positions,g)
    a = pi/(4*(1+float(g)))
    assert abs(y[-1]) < 1e-14
    assert abs(y[m-3] - (1+float(g))*sin(a)**2) < 1e-12


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_arbitrarily_small_rational_positive_g_is_proved(problem):
    g = Q(1,2**200)
    p = [Q(0),Q(1),Q(0)]
    cert = certify_positive_g_global_pf(
        p,g=g,problem=problem,k=4
    )
    assert cert.conclusion == "rigorously_global_pf_member"
    assert replay_positive_g_global_pf(
        p,cert,g=g,problem=problem,k=4
    )


@pytest.mark.parametrize("g", [Q(0),Q(1,10),Q(1,2),Q(1)])
def test_exact_support_lower_bound_increases_as_g_decreases(g):
    # Numerical cross-check only. The symbolic monotonicity proof in gt.md
    # and module docstring is the basis for global nondominance.
    if not g:
        return
    for part in [Q(1,2),Q(3,4),Q(99,100)]:
        gp = g*part
        a = pi/(4*(1+float(g)))
        ap = pi/(4*(1+float(gp)))
        bound = (1+float(g))*sin(a)**2
        smaller_g_bound = (1+float(gp))*sin(ap)**2
        assert smaller_g_bound > bound


@pytest.mark.parametrize("positions", [
    [Q(1,5),Q(1),Q(0)],
    [Q(0),Q(3,4),Q(0)],
    [Q(0),Q(1),Q(1,4)],
    [Q(0),Q(1,2),Q(1,2)],
])
def test_not_a_general_pareto_membership_oracle(positions):
    result = certify_positive_g_global_pf(
        positions,g=Q(1,2),problem="DTLZ5",k=4
    )
    assert result.conclusion == "unverified"
    assert not replay_positive_g_global_pf(
        positions,result.to_dict(),g=Q(1,2),
        problem="DTLZ5",k=4
    )


def test_replay_rejects_forged_proof_or_wrong_problem():
    p = [Q(0),Q(1),Q(0),Q(1,2)]
    parameters = dict(g=Q(1,2),problem="DTLZ5",k=4)
    record = certify_positive_g_global_pf(p,**parameters).to_dict()
    assert replay_positive_g_global_pf(p,record,**parameters)
    tampered = deepcopy(record)
    tampered["active_positions"][1]["numerator_hex"] = "0x0"
    assert not replay_positive_g_global_pf(p,tampered,**parameters)
    assert not replay_positive_g_global_pf(
        p,record,g=Q(1,3),problem="DTLZ5",k=4
    )
    assert not replay_positive_g_global_pf(
        p,record,g=Q(1,2),problem="DTLZ6",k=4
    )


@pytest.mark.parametrize("positions,g,problem,k", [
    ([Q(0),Q(1)],Q(1,2),"DTLZ5",4),
    ([Q(0),Q(1),Q(0),Q(-1,2)],Q(1,2),"DTLZ5",4),
    ([Q(0),Q(1),Q(0),Q(2)],Q(1,2),"DTLZ5",4),
    ([Q(0),Q(1),Q(0)],Q(-1,2),"DTLZ5",4),
    ([Q(0),Q(1),Q(0)],Q(2),"DTLZ5",4),
    ([Q(0),Q(1),Q(0)],Q(1,2),"DTLZ1",4),
    ([Q(0),Q(1),Q(0)],Q(1,2),"DTLZ5",0),
])
def test_rejects_invalid_input(positions,g,problem,k):
    with pytest.raises(ValueError):
        certify_positive_g_global_pf(
            positions,g=g,problem=problem,k=k
        )
