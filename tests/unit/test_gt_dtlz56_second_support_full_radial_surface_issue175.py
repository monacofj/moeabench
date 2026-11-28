# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive proofs for the second-support complete-radial PF surface."""
from fractions import Fraction
import pytest
from scripts.audit.dtlz56_second_support_full_radial_surface import (
    certify_second_support_full_radial_surface,
    replay_second_support_full_radial_surface,
)


def rational(v):
    return Fraction(int(v["numerator_hex"],16),int(v["denominator_hex"],16))


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_full_radial_second_surface_uses_independent_global_support(problem):
    r=certify_second_support_full_radial_surface(problem=problem)
    assert replay_second_support_full_radial_surface(r,problem=problem)
    assert r["complete_admissible_original_g_range"]
    assert len(r["grid_decisions"])==25
    assert rational(r["global_pf_uniform_margin"])>0
    assert rational(r["continuous_to_grid_raw_l2_upper"])>0
    assert r["second_support_objective"]==2
    assert r["sine_power"]==2
    assert all(rational(node["decision"][1])==1 and
               rational(node["decision"][2])==0 and
               node["supporting_objective"]==2
               for node in r["grid_decisions"])
    assert r["all_other_pf_angular_regions"]=="unverified"


def test_dtlz5_both_tail_branches_and_dtlz6_exact_tenth_power():
    five=certify_second_support_full_radial_surface(problem="DTLZ5")
    assert {rational(x["decision"][3]) for x in five["grid_decisions"]}=={
        Fraction(i,4) for i in range(5)}
    six=certify_second_support_full_radial_surface(problem="DTLZ6")
    assert all(rational(x["decision"][3])==
               rational(x["radial_g"])**10 for x in six["grid_decisions"])


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_grid_refinement_and_tampering(problem):
    base=certify_second_support_full_radial_surface(problem=problem)
    xmax=Fraction(1,4) if problem=="DTLZ5" else Fraction(3,16)
    fine=certify_second_support_full_radial_surface(
        problem=problem,
        x1_grid=tuple(xmax*i/8 for i in range(9)),
        radial_grid=tuple(Fraction(i,8) for i in range(9)))
    assert rational(fine["continuous_to_grid_raw_l2_upper"])<rational(
        base["continuous_to_grid_raw_l2_upper"])
    forged=dict(base)
    forged["global_pf_uniform_margin"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_second_support_full_radial_surface(
        forged,problem=problem)


def test_invalid_configuration_fails_closed():
    with pytest.raises(ValueError):
        certify_second_support_full_radial_surface(problem="DTLZ7")
    with pytest.raises(ValueError):
        certify_second_support_full_radial_surface(
            problem="DTLZ5",radial_grid=(0.,.5,.5,1.))
