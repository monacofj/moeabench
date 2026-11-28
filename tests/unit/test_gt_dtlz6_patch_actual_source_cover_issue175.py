# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New conservative DTLZ6 actual-source irrational-radius coverage tests."""
from fractions import Fraction
import pytest

from scripts.audit.dtlz6_patch_actual_source_cover import (
    certify_dtlz6_patch_to_source_cover,
    replay_dtlz6_patch_to_source_cover,
    _outward_original_objectives,
)
from scripts.audit.dtlz56_rational_interval import _pi_rational_bounds
from scripts.audit.dtlz5_patch_actual_source_cover import _decode


def exact_grid_decisions():
    return [(x,1.,1.,float(Fraction(u)**10))
            for x in (0.,1/16,1/8)
            for u in (0.,1/8,1/4)]


def test_original_dtlz6_rational_root_grid_to_actual_source():
    ps=exact_grid_decisions()
    r=certify_dtlz6_patch_to_source_cover(ps)
    assert replay_dtlz6_patch_to_source_cover(ps,r)
    assert len(r["per_grid_node_nearest_actual_source"])==9
    assert r["selected_decision_count"]==9
    synthetic=_decode(r["continuous_patch_to_grid_l2_upper"])
    actual=_decode(r["continuous_patch_to_actual_source_l2_upper"])
    assert synthetic<=actual<synthetic+Fraction(1,10**8)
    assert r["whole_gt_coverage"]=="unverified"


def test_irrational_tenth_root_and_cosine_endpoint_enclosed():
    pi=_pi_rational_bounds(16)
    for tail in (Fraction(1,2),Fraction(1,3)):
        intervals=_outward_original_objectives(
            (Fraction(1),Fraction(1,2),Fraction(1,2),tail),
            pi,root_bits=56)
        assert len(intervals)==4
        assert all(0<=lo<=hi for lo,hi in intervals)


def test_unrelated_source_gives_finite_conservative_bound():
    ps=[(0.,.5,.5,0.),(.5,.5,.5,.2),(1.,.5,.5,1.)]
    proof=certify_dtlz6_patch_to_source_cover(ps)
    assert replay_dtlz6_patch_to_source_cover(ps,proof)
    assert (_decode(proof["continuous_patch_to_actual_source_l2_upper"])
            >_decode(proof["continuous_patch_to_grid_l2_upper"]))


def test_forged_record_or_different_source_fails_closed():
    ps=exact_grid_decisions()
    proof=certify_dtlz6_patch_to_source_cover(ps)
    forged=dict(proof)
    forged["worst_grid_to_source_l1_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_dtlz6_patch_to_source_cover(ps,forged)
    ps[0]=(0.01,1.,1.,0.)
    assert not replay_dtlz6_patch_to_source_cover(ps,proof)


@pytest.mark.parametrize("bad", [
    [],[(0.,0.,0.)],[(0.,0.,0.,-1.)],
    [(0.,0.,0.,float("nan"))],
])
def test_invalid_reference_fails_closed(bad):
    with pytest.raises(ValueError):
        certify_dtlz6_patch_to_source_cover(bad)
