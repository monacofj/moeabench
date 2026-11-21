# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #119: new, independent and non-mutating GT inventory controls.

Necessary properties are tested separately from claims about global Pareto
membership or continuous-front completeness. No historical regression
test is modified or weakened.
"""
import numpy as np
import pytest

from scripts.audit.gt_remaining_mops_issue119 import (
    _MOPS, _independent_reference, audit,
)


@pytest.mark.parametrize("name",list(_MOPS))
def test_remaining_mops_audit_replays_and_respects_shapes(name):
    result=audit(name,3,64,probe_count=256,seed=11921)
    assert result["requested"]==64
    assert result["PS_rows"]>0
    assert result["PF_rows"]>0
    assert 0<result["PF_float64_exact_unique"]<=result["PF_rows"]
    assert result["PF_duplicate_rows"]==(
        result["PF_rows"]-result["PF_float64_exact_unique"])
    assert result["PS_outside_box"]==0
    assert result["repeat_exact"]
    assert not result["certifies_global_Pareto_membership"]
    assert not result["certifies_continuous_coverage"]
    if result["PS_PF_evaluation_disagreement_max"] is not None:
        assert result["PS_PF_evaluation_disagreement_max"]<1e-12


@pytest.mark.parametrize("name",["DTLZ2","DTLZ3","DTLZ4"])
@pytest.mark.parametrize("m",[3,5])
def test_spherical_reference_has_both_mathematical_geometry_and_exact_axis_anchors(name,m):
    r=audit(name,m,128,probe_count=512,seed=11932)
    assert r["PF_rows"]==128
    assert r["PF_float64_exact_unique"]==128
    assert r["unit_sphere_norm_max_error"]<1e-12
    assert r["axis_extreme_max_gap"]<3e-12
    assert r["independent_objective_probe"]["finite_probes_are_not_upper_bounds"]


@pytest.mark.parametrize("name,m",[("DTLZ1",3),("DTLZ1",5)])
def test_dtlz1_simplex_membership_and_missing_exact_vertex_are_separate(name,m):
    r=audit(name,m,128,probe_count=512,seed=11921)
    assert r["simplex_sum_max_error"]<1e-12
    # A random decision-space sample does not guarantee exact simplex
    # extremes: even a valid PF sample may leave a poor reference.
    assert r["simplex_extreme_max_gap"]>0
    assert r["PF_unique_internal_dominated"]==0


@pytest.mark.parametrize("name",["DTLZ7","DPF2"])
def test_disconnected_record_region_sampling_is_deterministic_and_necessary_valid(name):
    r=audit(name,5,256,d=3,probe_count=256)
    assert r["record_domain"]["outside_record_domain"]==0
    assert r["record_domain"]["valid_record_decisions"]==256
    assert r["record_domain"]["sampled_components"]>=2
    assert r["record_distance_variables_max_abs"]==0
    assert r["PF_unique_internal_dominated"]==0


@pytest.mark.parametrize("name",["DTLZ8","DTLZ9"])
def test_constrained_front_reports_actual_feasibility_without_conflating_it_with_optimality(name):
    r=audit(name,3,128,probe_count=256)
    assert r["constraints"]["G_shape"][0]==r["PS_rows"]
    assert r["constraints"]["G_shape"][1]==_MOPS[name](M=3).get_n_ieq_constr()
    assert r["constraints"]["positive_G_max"]>=0
    assert r["constraints"]["constraint_feasibility_is_not_global_Pareto_membership"]


@pytest.mark.parametrize("d,m",[(2,3),(3,5)])
def test_dpf1_essential_simplex_even_when_full_objectives_are_redundant(d,m):
    r=audit("DPF1",m,128,d=d,probe_count=256)
    assert r["essential_simplex_sum_max_error"]<1e-12
    assert r["PF_unique_internal_dominated"]==0


def test_dpf4_unsorted_thresholds_are_observed_not_silently_corrected():
    r=audit("DPF4",5,128,d=2,probe_count=256)
    assert "raw_chaos_thresholds_increasing" in r
    assert r["squared_projection_requires_paper_verification"]
    assert r["PS_rows"]==128


def test_independent_reference_distribution_obeys_simplex_and_sphere_geometry():
    simplex=_independent_reference("DTLZ1",5,128,11921)
    sphere=_independent_reference("DTLZ4",5,128,11921)
    np.testing.assert_allclose(simplex.sum(axis=1),.5,atol=1e-14,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(sphere,axis=1),1,atol=1e-14,rtol=0)
    assert _independent_reference("DTLZ9",5,128,11921) is None


@pytest.mark.parametrize("name",["DTLZ1","DTLZ2","DPF1","DTLZ9"])
def test_reject_unsupported_requested_audit_budget(name):
    with pytest.raises(ValueError):
        audit(name,3,1)
