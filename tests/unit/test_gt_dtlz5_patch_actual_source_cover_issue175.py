# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive exact patch-to-actual-PS bound and record replay tests."""
from fractions import Fraction

import pytest

from scripts.audit.dtlz5_patch_actual_source_cover import (
    certify_dtlz5_patch_to_source_cover,
    replay_dtlz5_patch_to_source_cover,
    _decode,
)


def source_grid():
    return [
        (x,1.,1.,t)
        for x in (0.,1/16,1/8)
        for t in (.5,.625,.75)
    ]


def test_actual_decisions_rigorously_cover_continuous_patch():
    decisions=source_grid()
    result=certify_dtlz5_patch_to_source_cover(decisions)
    assert replay_dtlz5_patch_to_source_cover(decisions,result)
    assert result["selected_decision_count"]==9
    assert len(result["per_grid_node_nearest_actual_source"])==9
    cover=_decode(result["continuous_patch_to_actual_source_l2_upper"])
    grid=_decode(result["continuous_patch_to_grid_l2_upper"])
    assert grid<=cover<grid+Fraction(1,10**8)
    assert result["whole_gt_coverage"]=="unverified"
    assert result["finite_precision_objective_F_alignment"]=="unverified"


def test_unrelated_reference_has_conservative_finite_bound():
    decisions=[(0.,.5,.5,.5),(.5,.5,.5,.5),(1.,.5,.5,.5)]
    result=certify_dtlz5_patch_to_source_cover(decisions)
    assert replay_dtlz5_patch_to_source_cover(decisions,result)
    assert (_decode(result["continuous_patch_to_actual_source_l2_upper"])
            >_decode(result["continuous_patch_to_grid_l2_upper"]))


def test_record_tamper_wrong_source_and_missing_source_fail_closed():
    source=source_grid()
    proof=certify_dtlz5_patch_to_source_cover(source)
    forged=dict(proof)
    forged["worst_grid_to_source_l1_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_dtlz5_patch_to_source_cover(source,forged)
    source[0]=(0.01,1.,1.,.5)
    assert not replay_dtlz5_patch_to_source_cover(source,proof)
    with pytest.raises(ValueError):
        certify_dtlz5_patch_to_source_cover([])


@pytest.mark.parametrize("bad",[
    [(0.,0.,0.)],
    [(0.,0.,0.,float("nan"))],
    [(0.,0.,0.,1.1)],
])
def test_invalid_source_decisions_rejected(bad):
    with pytest.raises(ValueError):
        certify_dtlz5_patch_to_source_cover(bad)
