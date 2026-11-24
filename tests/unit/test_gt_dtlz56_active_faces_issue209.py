# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive regression for strict original DTLZ5/6 active angular faces (#209).

No historical regression test or baseline is modified. In particular,
finite objective agreement does not substitute for a mathematically exact
boundary decision declaration when a global strict dominance theorem is
available.
"""
from dataclasses import replace
from math import pi, sin, cos

import numpy as np
import pytest

from moeabench.mops._dtlz56_front import (
    ScalarizedCandidate, decision_vector, scalarized_candidate,
)
from scripts.audit.dtlz56_paired_gt_evidence import _original_objectives
from scripts.audit.dtlz56_exact_same_ray import construct_same_ray_witness


def _upper_upper_candidate(g):
    r = 1 + g
    upper = pi * (1 + 2*g) / (4*r)
    theta = (pi/9, upper, upper)
    y = []
    for j in range(4):
        value = r
        for i in range(3-j):
            value *= cos(theta[i])
        if j:
            value *= sin(theta[3-j])
        y.append(value)
    return ScalarizedCandidate(
        g=g, scale=1.0, objectives=np.asarray(y),
        lower_bound=0.0, abs_gap=1e-7, nodes=3,
        angular_active_faces=(1,1),
    )


@pytest.mark.parametrize("problem,g", [
    ("DTLZ5", 0.03259761692867714),
    ("DTLZ6", 0.2),
])
def test_active_angular_faces_survive_inverse_rounding_exactly(problem,g):
    source = _upper_upper_candidate(g)
    # Make the spherical source objectives differ by a tiny amount:
    # this reproduces the inverse-angle numerical instability that
    # previously made an active endpoint strictly interior.
    noisy = source.objectives.copy()
    noisy[1] *= (1.0 + 1e-14)
    noisy[2] *= (1.0 + 1e-14)
    source = replace(source, objectives=noisy)
    decision = decision_vector(source,M=4,K=1,problem=problem)
    assert decision[1] == 1.0
    assert decision[2] == 1.0
    assert np.all(np.isfinite(decision))
    independently, _ = _original_objectives(
        decision[None,:], problem=problem,m=4,k=1,
    )
    assert np.max(np.abs(independently[0] - noisy)) < 5e-13

    # For original DTLZ5 the squared-distance radial g is exactly
    # rational from the stored tail, so a strict same-ray witness is
    # independently REJECTED once any inner position is exactly 1.
    if problem == "DTLZ5":
        from fractions import Fraction as Q
        x = tuple(Q.from_float(float(v)) for v in decision)
        exact_g = (x[-1] - Q(1,2))**2
        witness = construct_same_ray_witness(
            x[:3],g=exact_g,problem=problem,k=1,
        )
        assert witness.conclusion == "no_strict_same_ray_reduction"


def test_interior_and_historical_candidate_semantics_preserved():
    source = _upper_upper_candidate(.04)
    without_flags = replace(source,angular_active_faces=None)
    legacy = decision_vector(without_flags,M=4,K=1,problem="DTLZ5")
    assert legacy.shape == (4,)
    assert np.all((0 <= legacy) & (legacy <= 1))
    no_face = replace(source,angular_active_faces=(None,None))
    inferred = decision_vector(no_face,M=4,K=1,problem="DTLZ5")
    np.testing.assert_array_equal(inferred,legacy)


@pytest.mark.parametrize("flags",[
    (0,), (0,1,0), (2,1), ("upper",0),
])
def test_invalid_active_set_metadata_is_never_silently_trusted(flags):
    source = replace(_upper_upper_candidate(.04),
                     angular_active_faces=flags)
    with pytest.raises(ValueError,match="angular active faces"):
        decision_vector(source,M=4,K=1,problem="DTLZ5")


def test_real_scalarized_solver_exposes_original_active_set():
    weights = np.asarray([1.0,.1,.1,.05])
    result = scalarized_candidate(weights,.25,abs_tol=1e-5,max_nodes=5000)
    assert result.angular_active_faces is not None
    assert len(result.angular_active_faces)==2
    assert all(flag in (None,0,1) for flag in result.angular_active_faces)
    ps=decision_vector(result,M=4,K=1,problem="DTLZ5")
    for i,flag in enumerate(result.angular_active_faces, start=1):
        if flag is not None:
            assert ps[i] == float(flag)
    assert np.all((ps >= 0) & (ps <= 1))
