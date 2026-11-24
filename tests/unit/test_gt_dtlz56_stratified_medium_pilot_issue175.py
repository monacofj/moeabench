# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New tests for deterministic bounded source-ray intermediate diagnosis."""
import json

import numpy as np
import pytest

from scripts.audit import dtlz56_stratified_medium_pilot as medium


@pytest.mark.parametrize("problem", ["DTLZ5","DTLZ6"])
def test_ray_strata_match_exact_original_mechanism_and_are_reproducible(problem):
    mop, rows=medium._source_weights(problem,m=4,k=1)
    _,repeat=medium._source_weights(problem,m=4,k=1)
    assert len(rows)==24
    assert [z[:2] for z in rows]==[z[:2] for z in repeat]
    assert [z[0] for z in rows]==(
        ["sobol_boundary"]*8+["positive_gaussian"]*8+
        ["sparse_gamma"]*8
    )
    for (stratum,index,weights),(_,_,again) in zip(rows,repeat):
        np.testing.assert_array_equal(weights,again)
        assert weights.shape==(4,)
        assert np.all(np.isfinite(weights))
        assert np.all(weights>0)
        assert np.min(weights/weights.max())>=1e-7
    assert mop.N==4


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_small_real_source_ray_subset_is_audit_only(problem,monkeypatch):
    # One source ray from EACH family: the real production optimizer is
    # called only 3 times; the original full GT sampler is never run.
    monkeypatch.setattr(medium,"INDICES",{
        "sobol_boundary":(0,),
        "positive_gaussian":(0,),
        "sparse_gamma":(0,),
    })
    from moeabench.mops import DTLZ5,DTLZ6
    cls=DTLZ5 if problem=="DTLZ5" else DTLZ6
    def forbidden(*args,**kwargs):
        raise AssertionError("No expensive PS or full GT sampling")
    monkeypatch.setattr(cls,"ps",forbidden)
    report=medium.run_stratified_diagnostic(
        problem=problem,m=4,k=1,abs_tol=1e-4,max_nodes=2000,
    )
    assert report["stratified_ray_count"]==3
    assert len(report["optimizer"])==3
    assert report["independent_max_abs_error"]<1e-9
    assert report["exact_rational_g_audit"]["n_points"]==3
    assert all(x["abs_gap"]<=1.01e-4 for x in report["optimizer"])
    assert all(x["nodes"]<=2000 for x in report["optimizer"])
    assert all(x["count"]==1 for x in report["stratum_diagnostics"].values())
    assert report["full_gt_coverage"]=="unverified"
    assert report["scientific_certification"]=="unverified"
    if problem=="DTLZ6":
        assert len(report["certified_pf_proximity"])==3
        assert report["irrational_g_global_support"]["scientific_certification"]=="unverified"
    else:
        assert report["irrational_g_global_support"] is None
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize("kwargs",[
    {"problem":"DTLZ1"},
    {"problem":"DTLZ5","m":3},
    {"problem":"DTLZ5","k":2},
    {"problem":"DTLZ5","k":True},
])
def test_unsupported_medium_configuration_is_rejected_before_heavy_work(kwargs):
    with pytest.raises(ValueError):
        medium._source_weights(**kwargs)


@pytest.mark.parametrize("kwargs",[
    {"problem":"DTLZ5","abs_tol":0},
    {"problem":"DTLZ5","max_nodes":0},
    {"problem":"DTLZ5","max_nodes":12001},
    {"problem":"DTLZ5","max_nodes":True},
])
def test_budgets_must_be_explicitly_within_bounds(kwargs):
    with pytest.raises(ValueError):
        medium.run_stratified_diagnostic(**kwargs)


def test_production_mirror_expected_pool_is_not_full_selected_gt(monkeypatch):
    # Make sure the diagnostic never advertises the 24 preselected
    # directions as selected reference PS/F points.
    monkeypatch.setattr(medium,"INDICES",{
        "sobol_boundary":(0,),
        "positive_gaussian":(0,),
        "sparse_gamma":(0,),
    })
    report=medium.run_stratified_diagnostic(
        problem="DTLZ5",abs_tol=1e-4,max_nodes=2000,
    )
    assert report["method"]=="source_mirrored_preselection_stratified_scalarization"
    assert report["nominal_full_sampler_points"]==64
    assert report["stratified_ray_count"]==3
    assert report["full_gt_pareto_membership"]=="unverified"
