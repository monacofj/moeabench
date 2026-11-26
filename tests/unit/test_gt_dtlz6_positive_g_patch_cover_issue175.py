# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive exact DTLZ6 positive-g continuous subset and replay tests."""
from fractions import Fraction
import pytest

from scripts.audit.dtlz6_positive_g_patch_cover import (
    certify_dtlz6_positive_g_patch_cover,replay_dtlz6_positive_g_patch_cover,
)


def _fraction(field):
    return Fraction(int(field["numerator_hex"],16),
                    int(field["denominator_hex"],16))


def test_uniform_patch_all_exact_original_decisions_proved():
    proof=certify_dtlz6_positive_g_patch_cover()
    assert proof["conclusion"]=="rigorously_bounded_continuous_pf_patch"
    assert replay_dtlz6_positive_g_patch_cover(proof)
    assert _fraction(proof["uniform_pf_strict_margin"])>0
    assert len(proof["grid_decision_images"])==9
    assert sum(
        p["global_pf_membership_proof"]!="DTLZ56-UNIT-RADIUS"
        for p in proof["grid_decision_images"]
    )==6
    upper=_fraction(proof["raw_euclidean_continuous_patch_cover_upper"])
    assert Fraction(1,5)<upper<Fraction(1,4)
    assert proof["whole_gt_coverage"]=="unverified"


def test_refinement_improves_exact_subset_cover():
    coarse=certify_dtlz6_positive_g_patch_cover()
    fine=certify_dtlz6_positive_g_patch_cover(
        x1_grid=(0.,1/32,1/16,3/32,1/8),
        u_grid=(0.,1/16,1/8,3/16,1/4))
    assert _fraction(
        fine["raw_euclidean_continuous_patch_cover_upper"]
    )<_fraction(coarse["raw_euclidean_continuous_patch_cover_upper"])
    assert replay_dtlz6_positive_g_patch_cover(
        fine,x1_grid=(0.,1/32,1/16,3/32,1/8),
        u_grid=(0.,1/16,1/8,3/16,1/4))


def test_adversarial_record_rejected():
    proof=certify_dtlz6_positive_g_patch_cover()
    edited=dict(proof)
    edited["raw_euclidean_continuous_patch_cover_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_dtlz6_positive_g_patch_cover(edited)


@pytest.mark.parametrize("values",[
    (0.,1/8), (0.,1/16,1/16,1/8),
    (0.,float("nan"),1/8),(0.,1/8,1/16),
    (0.,1/16,1/4),
])
def test_invalid_first_angle_axis_fails_closed(values):
    with pytest.raises(ValueError):
        certify_dtlz6_positive_g_patch_cover(x1_grid=values)


def test_invalid_u_or_pi_fails_closed():
    with pytest.raises(ValueError):
        certify_dtlz6_positive_g_patch_cover(u_grid=(0.,.25,.125))
    with pytest.raises(ValueError):
        certify_dtlz6_positive_g_patch_cover(pi_terms=2)
