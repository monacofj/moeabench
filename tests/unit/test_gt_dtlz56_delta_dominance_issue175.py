# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-bound bounded-margin dominance exclusion (ADR 0059 / #175).

The oracle is independent of the existing GT sampler and only excludes
improvements of an explicit margin; it must never claim exact Pareto
membership on the strength of finite floating-point numerics.
"""
import json
import numpy as np
import pytest

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_dominance_bounds import (
    _relaxed_patch_can_fit, audit_delta_dominance,
)


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_independent_margin_exclusion_for_proved_g0_endpoint(cls):
    mop = cls(M=4)
    y = mop.pf(n_points=2)[0]
    audit = audit_delta_dominance(
        y, problem=cls.__name__, k=mop.K, margin=0.05,
        max_intervals=4096)
    assert audit.status == "excluded", audit.to_dict()
    assert audit.excluded_objectives == mop.M
    assert audit.intervals_unresolved == 0
    json.dumps(audit.to_dict(), allow_nan=False)


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_dominated_point_is_not_incorrectly_excluded(cls):
    mop = cls(M=4)
    # This vector is dominated by a feasible g=0 vector at exactly
    # the same angular direction, so margin 0.02 is realizable.
    y = 1.1*mop.pf(n_points=2)[0]
    audit = audit_delta_dominance(
        y, problem=cls.__name__, k=mop.K, margin=.02,
        max_intervals=64)
    assert audit.status == "unverified", audit.to_dict()
    assert audit.excluded_objectives < mop.M


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_requested_audit_is_visible_in_public_assessment(cls):
    mop = cls(M=4)
    y = mop.pf(n_points=2)
    result = mop.assess_gt(
        reference=y, dominance_margin=.05,
        dominance_max_intervals=256, probes=16)
    assert result.validity.status == "unverified"
    assert result.dominance.status == "unverified"
    assert result.dominance.value["numerically_excluded"] == len(y)
    assert result.dominance.value["unresolved"] == 0
    assert result.dominance.value["margin"] == .05
    json.dumps(result.to_dict(), allow_nan=False)


def test_no_implicit_margin_audit_and_non_dtlz_does_not_falsely_certify():
    from moeabench.mops import DTLZ2
    mop = DTLZ2(M=3)
    sample = mop.pf(n_points=16)
    result = mop.assess_gt(reference=sample, probes=16)
    assert result.dominance.status == "not_applicable"
    with pytest.raises(ValueError, match="only original DTLZ5/6"):
        mop.assess_gt(reference=sample, probes=16, dominance_margin=.05)


def test_bounded_margin_report_is_explicit_and_sidecar_read_only(tmp_path):
    mop = DTLZ5(M=4)
    path = tmp_path / "gt.json"
    mop.calibrate(source_baseline=path, n_points=2)
    before = path.read_bytes()
    report = mop.report(
        show=False, markdown=False, dominance_margin=.05,
        dominance_max_intervals=256, probes=16)
    assert "dominance" in report
    assert "numerically_excluded" in report
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad", [
    {"margin": 0}, {"margin": np.nan}, {"max_intervals": 0},
])
def test_invalid_numeric_protocol_is_rejected(bad):
    y = DTLZ5(M=4).pf(n_points=2)[0]
    options = dict(problem="DTLZ5", k=10, margin=.05)
    options.update(bad)
    with pytest.raises(ValueError):
        audit_delta_dominance(y, **options)


def test_optimistic_interval_relaxation_keeps_known_feasible_corner():
    # At g=0, the positive-sphere g=0 sample must lie in the relaxed
    # feasible patch; excluding it would create a false certificate.
    y = DTLZ5(M=4).pf(n_points=2)[0]
    assert _relaxed_patch_can_fit(y, 1., 0., slack=1e-11)


def test_insufficient_interval_budget_never_returns_a_false_exclusion():
    y = 1.1*DTLZ5(M=4).pf(n_points=2)[0]
    result = audit_delta_dominance(
        y, problem="DTLZ5", k=10, margin=.02, max_intervals=1)
    assert result.status == "unverified"
    assert result.intervals_unresolved > 0
