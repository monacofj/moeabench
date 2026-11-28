# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive full-radial second-support global PF curve proof tests."""
from fractions import Fraction
import pytest

from scripts.audit.dtlz56_second_support_full_radial import (
    certify_second_support_full_radial,replay_second_support_full_radial,
)


def decode(x):
    return Fraction(int(x["numerator_hex"],16),
                    int(x["denominator_hex"],16))


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_complete_second_support_face_and_proof_replay(problem):
    r=certify_second_support_full_radial(problem=problem)
    assert replay_second_support_full_radial(r,problem=problem)
    assert r["complete_admissible_g_range"] is True
    assert len(r["grid_decisions"])==9
    assert all(p["certificate"]["supporting_objective"]==2
               and p["certificate"]["sine_power"]==2
               for p in r["grid_decisions"])
    assert decode(r["raw_l2_continuous_to_grid_upper"])<Fraction(1,5)
    assert r["whole_gt_coverage"]=="unverified"


def test_dtlz5_both_tail_branches_and_dtlz6_exact_tenth_powers():
    d5=certify_second_support_full_radial(problem="DTLZ5")
    assert set(decode(p["radial_g"]) for p in d5["grid_decisions"])=={
        (Fraction(i,8)-Fraction(1,2))**2 for i in range(9)
    }
    d6=certify_second_support_full_radial(problem="DTLZ6")
    assert all(decode(p["decision"][-1])==decode(p["radial_g"])**10
               for p in d6["grid_decisions"])


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_finer_grid_reduces_exact_curve_cover(problem):
    coarse=certify_second_support_full_radial(
        problem=problem,radial_grid=(0.,.5,1.))
    fine=certify_second_support_full_radial(
        problem=problem,radial_grid=(0.,.25,.5,.75,1.))
    assert decode(fine["raw_l2_continuous_to_grid_upper"])<decode(
        coarse["raw_l2_continuous_to_grid_upper"])


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_forged_bounds_fail_closed(problem):
    r=certify_second_support_full_radial(problem=problem)
    bogus=dict(r)
    bogus["raw_l2_continuous_to_grid_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_second_support_full_radial(bogus,problem=problem)


@pytest.mark.parametrize("bad",[(),(0.,.5),(0.,.5,.5,1.),
                                (0.,.7,.4,1.),
                                (0.,float("nan"),1.)])
def test_bad_grid_rejected(bad):
    with pytest.raises(ValueError):
        certify_second_support_full_radial(
            problem="DTLZ5",radial_grid=bad)
