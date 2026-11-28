# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive research tests: rigorous continuous subset bounds, not full GT."""
from scripts.audit.dtlz56_proof_gated_continuous_coverage import (
    compare_continuous_support_coverage,
)
from scripts.audit.dtlz5_patch_actual_source_cover import _decode


def test_alternative_source_selection_has_independent_continuous_bounds():
    r=compare_continuous_support_coverage()
    assert r["schema"]==(
        "DTLZ56-ORIGINAL-M4K1-PROOF-GATED-CONTINUOUS-SUBSETS-REVIEW-V1")
    assert r["configuration"]=={
        "M":4,"K":1,"n_points":16,"original_scalarization_pool":448}
    for problem,dominated in (("DTLZ5",[10]),("DTLZ6",[7,11])):
        e=r[problem]
        assert e["same_first_four_gzero_anchors"]
        assert e["scientifically_known_bad_original_selected_indices"]==dominated
        assert len(e["proof_gated_pool_indices"])==12
        assert len(set(e["proof_gated_pool_indices"]))==12
        assert e["original_selected_sha256"]!=e["alternative_selected_sha256"]
        for label in ("original_declared_subset_coverage",
                      "alternative_declared_subset_coverage"):
            c=e[label]
            d=c["per_subset_rigorous_raw_l2_upper"]
            assert set(d)=={
                "complete_gzero_curve",
                "first_full_radial_support_surface",
                "second_full_radial_support_surface",
            }
            bounds=[_decode(v) for v in d.values()]
            assert all(v>=0 for v in bounds)
            assert _decode(c["union_rigorous_raw_l2_upper"])==max(bounds)
            assert c["entire_original_pf_coverage"]=="unverified"
            assert c["declared_subsets_only"] is True
        assert e["global_original_pf_exhaustiveness"]=="unverified"
        assert e["full_reference_IGD_IGDplus_HV_acceptance"]=="unverified"
    assert r["bounds_are_upper_not_exact_distances"] is True
    assert r["complete_original_pareto_front"]=="unverified"
    assert r["release_scientific_acceptance"]=="unverified"
    assert r["production_generator_modified"] is False
    assert r["accepted_adr_modified"] is False
    assert r["existing_regressions_modified"] is False
