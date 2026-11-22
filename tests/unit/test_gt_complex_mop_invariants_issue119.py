# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent read-only GT invariants tests; no historical regressions edited."""
import numpy as np
import pytest

from scripts.audit.gt_complex_mop_invariants_issue119 import audit, record_components
from moeabench.mops._dtlz7_pareto_regions import dtlz7_record_intervals


def test_independent_disconnected_record_membership_classifies_both_components():
    (a,b),(c,d)=dtlz7_record_intervals()
    x=np.array([[a,(a+b)/2],[(a+b)/2,(c+d)/2],
                [(c+d)/2,(a+b)/2],[(c+d)/2,(c+d)/2]])
    stats=record_components(x)
    assert stats["valid_parameter_rows"] == 4
    assert stats["observed_component_count"] == 4
    assert stats["expected_component_count"] == 4
    assert stats["component_completeness_certified"] is False
    y=record_components(np.array([[.4,.4]]))
    assert y["out_of_region_rows"]==1


@pytest.mark.parametrize("name",["DTLZ5","DTLZ6"])
def test_corrected_m5_reference_contains_nonzero_g_without_false_certificate(name):
    """The #119 finite audit observes #65's correction without overclaiming proof."""
    r=audit(name,5,32)
    assert r["known_incomplete_subfront_for_M_ge_4"] is False
    assert r["numerically_bounded_full_front_sampler"] is True
    assert r["distance_g_from_generated_decision_variables"] > 1e-6
    assert r["pf_returned"] == 32
    assert r["pf_duplicates"] == 0
    assert r["pf_internal_dominated"] == 0
    # This limited audit deliberately does not constitute a proof of
    # continuous front coverage; that requires the independent #65 method.
    assert r["globally_pareto_complete_certified"] is False
    assert not any("not a complete Ground Truth" in s for s in r["warnings"])


@pytest.mark.parametrize("name",["DTLZ7","DPF2"])
def test_record_domain_of_actual_mop_is_checked(name):
    r=audit(name,3,48)
    assert r["record_domain"]["out_of_region_rows"]==0
    assert r["globally_pareto_complete_certified"] is False
    assert r["record_domain"]["observed_component_count"]>0


@pytest.mark.parametrize("name",["DTLZ8","DTLZ9"])
def test_true_constraints_are_evaluated_independently(name):
    r=audit(name,3,48)
    assert r["constraints"]["constraint_count"] == (3 if name=="DTLZ8" else 2)
    assert r["constraints"]["feasibility_is_not_pareto_optimality"] is True
    assert r["pf_returned"] > 0


def test_dpf3_pf_filtering_is_observed_not_silently_counted_as_requested():
    r=audit("DPF3",3,48)
    assert r["pf_may_be_filtered"] is True
    assert r["pf_returned"] <= r["ps_returned"]
    assert r["pf_shortfall"]==48-r["pf_returned"]
    assert r["globally_pareto_complete_certified"] is False


def test_reject_invalid_intrinsic_d():
    with pytest.raises(ValueError,match="intrinsic dimension"):
        audit("DPF2",3,32,intrinsic_d=3)
