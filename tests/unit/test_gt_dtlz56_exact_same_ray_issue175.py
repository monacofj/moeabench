# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New proof-replay tests, without modifying the historical regression suite."""
from copy import deepcopy
from fractions import Fraction
from math import cos, sin, pi

import pytest

from scripts.audit.dtlz56_exact_same_ray import (
    _decode, construct_same_ray_witness, verify_same_ray_witness,
)


def _published_objectives(positions, g):
    """Independent trigonometric reconstruction of original DTLZ5/6."""
    m = len(positions)+1
    angles = [pi*float(positions[0])/2]
    angles += [pi*(1+2*g*float(x))/(4*(1+g)) for x in positions[1:]]
    out = []
    for j in range(m):
        val = 1+g
        for i in range(m-j-1):
            val *= cos(angles[i])
        if j:
            val *= sin(angles[m-j-1])
        out.append(val)
    return out


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [2,3,4,5,10])
@pytest.mark.parametrize("u", [Fraction(1,4), Fraction(1,2), Fraction(3,4)])
def test_exact_interior_ray_witness_is_replayable(problem, m, u):
    positions = [Fraction(1,3)]+[u]*(m-2)
    gy = Fraction(1,4)
    a = construct_same_ray_witness(
        positions,g=gy,problem=problem,k=4
    )
    assert a.conclusion == "exact_same_ray_dominator"
    assert verify_same_ray_witness(
        positions,a.to_dict(),g=gy,problem=problem,k=4
    )
    gp = _decode(a.witness_g)
    assert 0 <= _decode(a.minimum_ray_g) < gp < gy
    source = _published_objectives(positions,float(gy))
    target = _published_objectives(
        [_decode(x) for x in a.witness_positions],float(gp)
    )
    ratio = float(_decode(a.objective_scale))
    for before,after in zip(source,target):
        assert abs(after-ratio*before) < 1e-12
        assert after <= before+1e-12
    assert max(source) > max(target)


@pytest.mark.parametrize("edge", [Fraction(0), Fraction(1)])
def test_inner_angular_endpoint_is_not_falsely_certified(edge):
    a = construct_same_ray_witness(
        [Fraction(1,3),edge],g=Fraction(1,5),
        problem="DTLZ5",k=4
    )
    assert a.conclusion == "no_strict_same_ray_reduction"
    assert a.witness_g is None


def test_one_active_inner_endpoint_blocks_same_ray_only():
    a=construct_same_ray_witness(
        [Fraction(1,2), Fraction(1,3), Fraction(1)],
        g=Fraction(1,5),problem="DTLZ6",k=4)
    assert a.conclusion == "no_strict_same_ray_reduction"


@pytest.mark.parametrize("m", [2,3,4,5,10])
def test_zero_g_does_not_produce_false_dominator(m):
    a = construct_same_ray_witness(
        [Fraction(1,2)]*(m-1),g=Fraction(0),
        problem="DTLZ5",k=4
    )
    assert a.conclusion == "no_strict_same_ray_reduction"


def test_rational_near_endpoint_requires_no_numeric_gap():
    gy = Fraction(1,2**150)
    a = construct_same_ray_witness(
        [Fraction(1,2),Fraction(1,3)],g=gy,
        problem="DTLZ6",k=4,
    )
    assert a.conclusion == "exact_same_ray_dominator"
    assert 0 < _decode(a.witness_g) < gy
    assert verify_same_ray_witness(
        [Fraction(1,2), Fraction(1,3)],a.to_dict(),
        g=gy,problem="DTLZ6",k=4
    )


def test_tampered_witness_rejected():
    args = dict(g=Fraction(1,5),problem="DTLZ5",k=4)
    src = [Fraction(1,3),Fraction(1,4),Fraction(3,5)]
    record = construct_same_ray_witness(src,**args).to_dict()
    assert verify_same_ray_witness(src,record,**args)
    for key in ("objective_scale","witness_g","minimum_ray_g"):
        altered=deepcopy(record)
        altered[key]["numerator_hex"] = "0x0"
        assert not verify_same_ray_witness(src,altered,**args)


@pytest.mark.parametrize("positions,g,problem,k", [
    ([Fraction(1,2),-1],Fraction(1,5),"DTLZ5",4),
    ([Fraction(1,2),2],Fraction(1,5),"DTLZ5",4),
    ([],Fraction(1,5),"DTLZ5",4),
    ([Fraction(1,2)],-1,"DTLZ5",4),
    ([Fraction(1,2)],2,"DTLZ5",4),
    ([Fraction(1,2)],Fraction(1,4),"DTLZ6",0),
    ([Fraction(1,2)],Fraction(1,4),"DTLZ2",4),
]):
def test_invalid_inputs_rejected(positions,g,problem,k):
    with pytest.raises(ValueError):
        construct_same_ray_witness(positions,g=g,problem=problem,k=k)
