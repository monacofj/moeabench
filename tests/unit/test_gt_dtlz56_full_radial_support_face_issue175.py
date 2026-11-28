# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive full-radial continuous original PF face proof checks."""
from fractions import Fraction
import pytest
from scripts.audit.dtlz56_full_radial_support_face import (
    certify_full_radial_support_face,replay_full_radial_support_face,
)


def _rational(item):
    return Fraction(int(item["numerator_hex"],16),
                    int(item["denominator_hex"],16))


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_complete_radial_support_face_uniformly_global_pf(problem):
    proof=certify_full_radial_support_face(problem=problem)
    assert replay_full_radial_support_face(proof,problem=problem)
    assert proof["conclusion"]=="rigorously_global_pf_continuous_full_radial_face"
    assert proof["certified_full_original_radial_range"] is True
    assert proof["restricted_angular_support_face_only"] is True
    assert len(proof["grid_decisions"])==25
    assert _rational(proof["global_pf_strict_uniform_margin"])>0
    assert _rational(proof["continuous_to_exact_grid_raw_l2_upper"])>0
    assert proof["whole_positive_g_pf"]=="unverified"
    assert proof["scientific_certification"]=="unverified"


def test_both_signed_dtlz5_tail_branches_are_covered():
    proof=certify_full_radial_support_face(problem="DTLZ5")
    tails=set(_rational(row["decision"][-1])
              for row in proof["grid_decisions"])
    assert min(tails)==0 and max(tails)==1
    assert Fraction(1,2) in tails
    assert all(tail in tails for tail in
               (Fraction(1,4),Fraction(3,4)))


def test_dtlz6_grid_tail_is_exact_tenth_power():
    proof=certify_full_radial_support_face(problem="DTLZ6")
    for row in proof["grid_decisions"]:
        tail=_rational(row["decision"][-1])
        g=_rational(row["radial_g"])
        assert tail==g**10


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_finer_grid_reduces_certified_continuous_bound(problem):
    first=certify_full_radial_support_face(problem=problem)
    xhi=Fraction(1,4) if problem=="DTLZ5" else Fraction(3,16)
    refined=certify_full_radial_support_face(
        problem=problem,
        x1_grid=tuple(xhi*i/8 for i in range(9)),
        radial_grid=tuple(Fraction(i,8) for i in range(9)))
    assert _rational(
        refined["continuous_to_exact_grid_raw_l2_upper"]
    )<_rational(first["continuous_to_exact_grid_raw_l2_upper"])


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_forged_proof_rejected(problem):
    proof=certify_full_radial_support_face(problem=problem)
    forged=dict(proof)
    forged["global_pf_strict_uniform_margin"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_full_radial_support_face(forged,problem=problem)


def test_invalid_domain_fails_closed():
    with pytest.raises(ValueError):
        certify_full_radial_support_face(problem="DTLZ7")
    with pytest.raises(ValueError):
        certify_full_radial_support_face(
            problem="DTLZ5",radial_grid=(0.,.5,.5,1.))
    with pytest.raises(ValueError):
        certify_full_radial_support_face(problem="DTLZ6",pi_terms=1)
