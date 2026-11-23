# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent global-radius dominance theorem and bounded numerical audit.

No finite-g tolerance is presented as exact Pareto certification.
Existing GT regressions/baselines are not changed.
"""
import json

import numpy as np
import pytest

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_global_radius import (
    audit_global_radius_dominance,
)


def _manual_g0_endpoint(m):
    # Explicit original closed-form theta1=0; theta_i=pi/4.
    a = np.array(
        [2.0**(-(m-2)/2)] +
        [2.0**(-(m-j)/2) for j in range(2, m)]
        + [0.0]
    )
    np.testing.assert_allclose(np.linalg.norm(a), 1.0, atol=2e-15)
    return a


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_global_minimum_radius_of_g0_point_excludes_any_positive_gap(problem):
    y = _manual_g0_endpoint(4)
    result = audit_global_radius_dominance(
        y, problem=problem, k=10, g_gap=.01)
    assert result.status == "excluded"
    assert result.intervals_tested == 0
    assert result.g_candidate == pytest.approx(0.0, abs=1e-15)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_independently_constructed_strict_dominator_is_found(problem):
    z = _manual_g0_endpoint(4)
    y = 1.1 * z
    # y is independently known to be feasible at g=0.1: all inner
    # angles remain pi/4 and both original feasible g ranges include it.
    audit = audit_global_radius_dominance(
        y, problem=problem, k=10, g_gap=.02)
    assert audit.status == "witness", audit.to_dict()
    assert audit.witness_g is not None
    assert audit.witness_g <= .08 + 1e-9
    witness = np.array(audit.witness_objectives)
    assert np.all(witness <= y + 1e-10)
    assert np.linalg.norm(witness) < np.linalg.norm(y) - .01
    assert np.linalg.norm(witness) == pytest.approx(1+audit.witness_g, abs=1e-10)


@pytest.mark.parametrize("klass", [DTLZ5, DTLZ6])
def test_public_api_and_report_expose_same_read_only_radius_audit(klass, tmp_path):
    mop = klass(M=4)
    y = np.array([_manual_g0_endpoint(4)])
    outcome = mop.assess_gt(
        reference=y, probes=16, dominance_g_gap=.01,
        dominance_g_max_intervals=8)
    assert outcome.validity.status == "unverified"  # original full M>=4 PF
    assert outcome.global_dominance.status == "unverified"
    assert outcome.global_dominance.value["numerically_excluded"] == 1
    assert outcome.global_dominance.value["unresolved"] == 0
    json.dumps(outcome.to_dict(), allow_nan=False)

    sidecar = tmp_path / "gt.json"
    mop.calibrate(source_baseline=sidecar, n_points=2)
    original_bytes = sidecar.read_bytes()
    report = mop.report(
        show=False, markdown=False, dominance_g_gap=.01,
        dominance_g_max_intervals=8, probes=16)
    assert "global_dominance" in report
    assert "numerically_excluded" in report
    assert sidecar.read_bytes() == original_bytes


def test_bounded_global_check_never_claims_exact_membership():
    y = _manual_g0_endpoint(4)
    out = DTLZ5(M=4).assess_gt(reference=y[None, :], probes=16)
    assert out.global_dominance.status == "not_applicable"
    with pytest.raises(ValueError, match="only original DTLZ5/6"):
        from moeabench.mops import DTLZ2
        DTLZ2(M=4).assess_gt(
            reference=y[None, :], probes=16, dominance_g_gap=.01)


def test_unresolved_interval_budget_cannot_falsely_exclude(monkeypatch):
    import moeabench.mops._dtlz56_global_radius as oracle
    monkeypatch.setattr(oracle, "_relaxed_patch_can_fit",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(oracle, "_fixed_g_witness",
                        lambda *args, **kwargs: None)
    y = 1.1*_manual_g0_endpoint(4)
    result = oracle.audit_global_radius_dominance(
        y, problem="DTLZ5", k=10, g_gap=.01, max_intervals=1)
    assert result.status == "unverified"
    assert result.intervals_unresolved > 0


@pytest.mark.parametrize("invalid", [
    {"g_gap": 0},
    {"g_gap": np.nan},
    {"max_intervals": 0},
    {"numerical_slack": 0},
])
def test_numerical_protocol_rejects_invalid_parameters(invalid):
    y = _manual_g0_endpoint(4)
    args = dict(problem="DTLZ5", k=10, g_gap=.01)
    args.update(invalid)
    with pytest.raises(ValueError):
        audit_global_radius_dominance(y, **args)


def test_independent_global_oracle_does_not_reuse_production_scalarization(
        monkeypatch):
    import moeabench.mops._dtlz56_front as production
    monkeypatch.setattr(
        production, "scalarized_candidate",
        lambda *args, **kwargs: pytest.fail("production solver reused"))
    out = audit_global_radius_dominance(
        1.1*_manual_g0_endpoint(4),
        problem="DTLZ5", k=10, g_gap=.01)
    assert out.status == "witness"
