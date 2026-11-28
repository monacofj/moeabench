# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive fail-closed three-certified-subset union ledger tests."""
import pytest

from scripts.audit.dtlz56_three_subset_union_cover import (
    certify_three_subset_union,replay_three_subset_union,
)
from scripts.audit.dtlz56_full_radial_face_actual_source import (
    certify_full_radial_face_to_source,
)
from scripts.audit.dtlz56_second_support_actual_source import (
    certify_second_support_to_actual_ps,
)
from scripts.audit.dtlz5_patch_actual_source_cover import _decode


@pytest.fixture(scope="module")
def dtlz5_proof_bundle():
    ps=[(0.,.5,.5,.5),(1/3,.5,.5,.5),
        (2/3,.5,.5,.5),(1.,.5,.5,.5)]
    return (ps,
            certify_full_radial_face_to_source(ps,problem="DTLZ5"),
            certify_second_support_to_actual_ps(ps,problem="DTLZ5"))


def test_union_is_maximum_of_three_replayed_exact_bounds(dtlz5_proof_bundle):
    ps,full,second=dtlz5_proof_bundle
    record=certify_three_subset_union(
        ps,problem="DTLZ5",
        full_radial_report=full,second_support_report=second)
    assert replay_three_subset_union(
        ps,record,problem="DTLZ5",
        full_radial_report=full,second_support_report=second)
    vals=[_decode(v) for v in record["continuous_component_upper_bounds"].values()]
    assert len(vals)==3
    assert _decode(record["union_continuous_cover_raw_l2_upper"])==max(vals)
    assert record["certified_subset_count"]==3
    assert record["whole_pareto_front_cover"]=="unverified"
    assert record["igd_igdplus_hv_suitability"]=="unverified"
    assert record["scientific_certification"]=="unverified"


def test_forged_union_or_distinct_source_fails_closed(dtlz5_proof_bundle):
    ps,full,second=dtlz5_proof_bundle
    record=certify_three_subset_union(
        ps,problem="DTLZ5",
        full_radial_report=full,second_support_report=second)
    forged=dict(record)
    forged["union_continuous_cover_raw_l2_upper"]={
        "numerator_hex":"0x0","denominator_hex":"0x1"}
    assert not replay_three_subset_union(
        ps,forged,problem="DTLZ5",
        full_radial_report=full,second_support_report=second)
    wrong=list(ps)
    wrong[0]=(0.01,.5,.5,.5)
    assert not replay_three_subset_union(
        wrong,record,problem="DTLZ5",
        full_radial_report=full,second_support_report=second)


def test_cross_problem_reference_rejected(dtlz5_proof_bundle):
    ps,full,second=dtlz5_proof_bundle
    with pytest.raises(ValueError):
        certify_three_subset_union(
            ps,problem="DTLZ6",
            full_radial_report=full,second_support_report=second)
