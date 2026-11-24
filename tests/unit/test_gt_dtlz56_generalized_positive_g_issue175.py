# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent generalized global-membership proof tests (#175)."""
from copy import deepcopy
from fractions import Fraction as Q
from math import cos, pi, sin

import pytest

from scripts.audit.dtlz56_generalized_positive_g_pf import (
    certify_generalized_positive_g_pf,
    replay_generalized_positive_g_pf,
)


def original_objectives(u, g):
    """Published original DTLZ5/6 spherical equations, independently."""
    m = len(u)+1
    angles = [pi*float(u[0])/2]
    angles += [pi*(1+2*float(g)*float(x))/(4*(1+float(g)))
               for x in u[1:]]
    result = []
    for j in range(m):
        v = 1+float(g)
        for index in range(m-j-1):
            v *= cos(angles[index])
        if j:
            v *= sin(angles[m-j-1])
        result.append(v)
    return result


def boundary_positions(m, j):
    """Construct the exact supporting stratum for j=1,...,M-2."""
    u = [Q(1,3)]*(m-1)
    u[0] = 0
    if j == 1:
        for index in range(1,m-1):
            u[index] = 1
    else:
        sine_index = m-j
        for index in range(1,sine_index):
            u[index] = 1
        u[sine_index] = 0
    return u


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [4,5,7,10])
@pytest.mark.parametrize("g", [Q(0),Q(1,8),Q(1,2),Q(1)])
def test_each_supporting_objective_proves_global_membership(problem,m,g):
    for j in range(1,m-1):
        u = boundary_positions(m,j)
        result = certify_generalized_positive_g_pf(
            u,g=g,problem=problem,k=4)
        assert result.conclusion == "rigorously_global_pf_member"
        assert result.supporting_objective == j
        power = m-2 if j == 1 else m-j
        assert result.sine_power == power
        assert power >= 2
        assert result.to_dict()["scientific_certification"] == "unverified"
        assert replay_generalized_positive_g_pf(
            u,result.to_dict(),g=g,problem=problem,k=4)
        y = original_objectives(u,g)
        a = pi/(4*(1+float(g)))
        assert abs(y[-1]) < 1e-14
        assert abs(y[j-1]-(1+float(g))*sin(a)**power) < 1e-11


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_original_m4_theorem_is_preserved(problem):
    result = certify_generalized_positive_g_pf(
        [Q(0),Q(1),Q(0)],g=Q(1,3),problem=problem,k=4)
    assert result.supporting_objective == 2
    assert result.sine_power == 2


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_exact_small_positive_radius_with_high_power(problem):
    u = boundary_positions(10,1)
    g = Q(1,2**200)
    cert = certify_generalized_positive_g_pf(
        u,g=g,problem=problem,k=4)
    assert cert.conclusion == "rigorously_global_pf_member"
    assert cert.supporting_objective == 1
    assert cert.sine_power == 8
    assert replay_generalized_positive_g_pf(
        u,cert,g=g,problem=problem,k=4)


@pytest.mark.parametrize("m", [4,5,7,10])
def test_independent_angular_samples_obey_support_lower_bound(m):
    # These finite samples check the algebraic objective-index mapping;
    # the strict global claim rests on the separately proved theorem.
    for g in (0.0,0.2,1.0):
        a = pi/(4*(1+g))
        b = pi/2-a
        for j in range(1,m-1):
            p = m-2 if j == 1 else m-j
            lower = (1+g)*sin(a)**p
            for blend in (0.0,0.25,0.5,0.75,1.0):
                theta = [0]+[a+(b-a)*blend]*(m-2)
                value=1+g
                for index in range(m-j):
                    value *= cos(theta[index])
                if j>1:
                    value *= sin(theta[m-j])
                assert value >= lower-1e-12


@pytest.mark.parametrize("u",[
    [Q(1,2),Q(1),Q(1),Q(1)],
    [Q(0),Q(3,4),Q(1),Q(1)],
    [Q(0),Q(1),Q(1),Q(1,3)],
    [Q(0),Q(1),Q(1,4),Q(1)],
])
def test_off_stratum_does_not_claim_global_pf(u):
    result = certify_generalized_positive_g_pf(
        u,g=Q(1,2),problem="DTLZ5",k=4)
    assert result.conclusion == "unverified"
    assert not replay_generalized_positive_g_pf(
        u,result,g=Q(1,2),problem="DTLZ5",k=4)


def test_forgery_and_replay_mismatch_rejected():
    u = boundary_positions(7,3)
    args=dict(g=Q(1,2),problem="DTLZ5",k=4)
    original=certify_generalized_positive_g_pf(u,**args).to_dict()
    assert replay_generalized_positive_g_pf(u,original,**args)
    for field in ("supporting_objective","sine_power","tail_dimension"):
        forged = deepcopy(original)
        forged[field] = 0
        assert not replay_generalized_positive_g_pf(u,forged,**args)
    forged=deepcopy(original)
    forged["position_declaration"][0]["numerator_hex"]="0x1"
    assert not replay_generalized_positive_g_pf(u,forged,**args)
    assert not replay_generalized_positive_g_pf(
        u,original,g=Q(1,3),problem="DTLZ5",k=4)


@pytest.mark.parametrize("u,g,problem,k",[
    ([Q(0),Q(1)],Q(1,2),"DTLZ5",4),
    ([Q(0),Q(1),Q(1),Q(2)],Q(1,2),"DTLZ5",4),
    ([Q(0),Q(1),Q(1),Q(1)],Q(-1,2),"DTLZ5",4),
    ([Q(0),Q(1),Q(1),Q(1)],Q(2),"DTLZ5",4),
    ([Q(0),Q(1),Q(1),Q(1)],Q(1,2),"DTLZ2",4),
    ([Q(0),Q(1),Q(1),Q(1)],Q(1,2),"DTLZ5",0),
])
def test_invalid_original_definition_rejected(u,g,problem,k):
    with pytest.raises(ValueError):
        certify_generalized_positive_g_pf(
            u,g=g,problem=problem,k=k)
