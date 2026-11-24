# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New exact-rational boundary tangent proofs; no historical regression edits."""
from copy import deepcopy
from fractions import Fraction as Q
from math import cos, pi, sin

import pytest

from scripts.audit.dtlz56_rational_tangent import (
    certify_rational_tangent_descent,
    replay_rational_tangent_certificate,
    _trigonometric_bounds,
)


def _published_objectives(u, g):
    """Independent original hyperspherical equations (finite sanity check)."""
    m = len(u)+1
    angles = [pi*float(u[0])/2]+[
        pi*(1+2*float(g)*float(pos))/(4*(1+float(g)))
        for pos in u[1:]
    ]
    output = []
    for j in range(m):
        value = 1+float(g)
        for i in range(m-j-1):
            value *= cos(angles[i])
        if j:
            value *= sin(angles[m-j-1])
        output.append(value)
    return output


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("positions,velocity", [
    ([Q(1,2),Q(0),Q(1,2)], [Q(-1,4),Q(0),Q(7,4)]),
    ([Q(1,2),Q(1),Q(1)], [Q(-2,5),Q(0),Q(0)]),
    ([Q(1,2),Q(0),Q(0)], [Q(-1,4),Q(0),Q(0)]),
])
def test_boundary_different_ray_proofs(problem,positions,velocity):
    args = dict(g=Q(1,2),velocity=velocity,problem=problem,k=4,
                pi_terms=6,trig_pairs=2)
    result = certify_rational_tangent_descent(positions,**args)
    assert result.conclusion == "rigorous_local_dominator", result.to_dict()
    assert replay_rational_tangent_certificate(
        positions,result.to_dict(),**args)
    # Independent finite numerical check of the exact derivative theorem,
    # not the basis of the rational proof.
    eps = Q(1,100000)
    before = _published_objectives(positions,args["g"])
    after = _published_objectives([
        x-eps*v for x,v in zip(positions,velocity)
    ],args["g"]-eps)
    assert all(b>a for a,b in zip(after,before))


def test_non_strict_derivative_abstains():
    positions = [Q(0), Q(0), Q(1)]
    result = certify_rational_tangent_descent(
        positions,g=Q(1,2),velocity=[Q(0)]*3,
        problem="DTLZ6",k=4,
    )
    assert result.conclusion == "unverified"
    assert not replay_rational_tangent_certificate(
        positions,result.to_dict(),g=Q(1,2),velocity=[Q(0)]*3,
        problem="DTLZ6",k=4,
    )


def test_forged_derivative_or_velocity_is_not_replayable():
    positions = [Q(1,2),Q(1),Q(1)]
    options = dict(g=Q(1,2),velocity=[Q(-2,5),Q(0),Q(0)],
                   problem="DTLZ5",k=4)
    original = certify_rational_tangent_descent(
        positions,**options).to_dict()
    assert replay_rational_tangent_certificate(positions,original,**options)
    altered = deepcopy(original)
    altered["derivative_intervals"][0]["lower"]["numerator_hex"] = "0x0"
    assert not replay_rational_tangent_certificate(
        positions,altered,**options)


def test_exact_trig_bounds_enclose_python_sine_and_cosine():
    from scripts.audit.dtlz56_rational_interval import _pi_rational_bounds
    a,b = _pi_rational_bounds(6)
    for portion in (Q(0), Q(1,8), Q(1,4), Q(1,2)):
        sine,cosine = _trigonometric_bounds(
            (a*portion,b*portion),2)
        assert float(sine[0])-1e-15 <= sin(pi*float(portion))
        assert sin(pi*float(portion)) <= float(sine[1])+1e-15
        assert float(cosine[0])-1e-15 <= cos(pi*float(portion))
        assert cos(pi*float(portion)) <= float(cosine[1])+1e-15


@pytest.mark.parametrize("positions,velocity,g,problem,k", [
    ([Q(1,2),Q(0)], [Q(0),Q(1)],Q(1,2),"DTLZ5",4),
    ([Q(1,2),Q(1)], [Q(0),Q(-1)],Q(1,2),"DTLZ5",4),
    ([Q(1,2),Q(0)], [Q(0)],Q(1,2),"DTLZ5",4),
    ([Q(-1,2),Q(0)], [Q(0),Q(0)],Q(1,2),"DTLZ5",4),
    ([Q(1,2),Q(0)], [Q(0),Q(0)],Q(0),"DTLZ5",4),
    ([Q(1,2),Q(0)], [Q(0),Q(0)],Q(2),"DTLZ5",4),
    ([Q(1,2),Q(0)], [Q(0),Q(0)],Q(1,2),"DTLZ2",4),
    ([Q(1,2),Q(0)], [Q(0),Q(0)],Q(1,2),"DTLZ6",0),
])
def test_bad_inputs_rejected(positions,velocity,g,problem,k):
    with pytest.raises(ValueError):
        certify_rational_tangent_descent(
            positions,g=g,velocity=velocity,problem=problem,k=k)
