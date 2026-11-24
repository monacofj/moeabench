# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive regression-independent checks of source-bound small pilot."""
import json

import numpy as np
import pytest

from scripts.audit import dtlz56_small_scalarization_pilot as pilot


@pytest.mark.parametrize("problem", ["DTLZ5","DTLZ6"])
def test_source_bound_pilot_consumes_real_scalarization_and_never_promotes_gt(
    problem,monkeypatch,
):
    monkeypatch.setattr(pilot,"RAYS",pilot.RAYS[:1])
    record=pilot.run_pilot(
        problem=problem,m=4,k=1,abs_tol=1e-4,max_nodes=2000,
    )
    assert record["ray_count"]==1
    assert record["method"]=="production_scalarized_candidate_and_decision_vector"
    assert record["scope"]=="fixed_preference_rays_not_full_gt_or_front_cover"
    assert record["independent_max_abs_objective_error"] < 1e-9
    assert record["per_row_audit"]["independent_objective_alignment"]=="pass"
    assert record["per_row_audit"]["n_points"]==1
    assert record["per_row_audit"]["whole_reference_membership"]=="unverified"
    assert record["full_gt_coverage"]=="unverified"
    assert record["scientific_certification"]=="unverified"
    assert len(record["optimizer"])==1
    assert np.isfinite(record["optimizer"][0]["scalarization_abs_gap"])
    assert record["optimizer"][0]["scalarization_abs_gap"]>0
    assert len(record["decision_bytes_sha256"])==64
    json.dumps(record,allow_nan=False)


@pytest.mark.parametrize("invalid",[
    {"problem":"DTLZ1","m":4,"k":1},
    {"problem":"DTLZ5","m":3,"k":1},
    {"problem":"DTLZ5","m":4,"k":0},
    {"problem":"DTLZ5","m":4,"k":True},
    {"problem":"DTLZ5","m":4,"k":1,"abs_tol":0},
    {"problem":"DTLZ5","m":4,"k":1,"max_nodes":0},
])
def test_invalid_pilot_arguments_rejected_without_sampling(invalid):
    with pytest.raises(ValueError):
        pilot.run_pilot(**invalid)


def test_pilot_cannot_be_silent_full_gt_sweep(monkeypatch):
    monkeypatch.setattr(pilot,"RAYS",pilot.RAYS[:1])
    class Forbidden:
        def __call__(self,*args,**kwargs):
            raise AssertionError("Full sampler must not run")
    from moeabench.mops import DTLZ5
    monkeypatch.setattr(DTLZ5,"ps",Forbidden())
    record=pilot.run_pilot(
        problem="DTLZ5",m=4,k=1,abs_tol=1e-4,max_nodes=2000,
    )
    assert record["ray_count"]==1
