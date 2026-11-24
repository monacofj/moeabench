# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive API coverage: objective-only independent DTLZ5/6 local audit.

All pre-existing regression and frozen reference tests remain untouched.
"""
from math import cos, pi, sin

import numpy as np
import pytest

import moeabench as mb


def _original_row(m=4, g=0.3):
    """Original DTLZ5/6 angular formula, without any production evaluator."""
    theta = [pi/4]+[pi/4]*(m-2)
    row = np.empty(m)
    for j in range(m):
        value = 1.0 + g
        for i in range(m-j-1):
            value *= cos(theta[i])
        if j:
            value *= sin(theta[m-j-1])
        row[j] = value
    return row


@pytest.mark.parametrize("klass", [mb.mops.DTLZ5, mb.mops.DTLZ6])
@pytest.mark.parametrize("m", [4, 5])
def test_objective_only_local_witness_in_public_assessment(klass, m):
    mop = klass(M=m)
    candidate = np.array([_original_row(m)])
    original = candidate.copy()
    report = mop.assess_gt(
        reference=candidate, probes=16, objective_only_local_descent=True
    )
    assert report.validity.status != "pass"
    assert report.local_descent.status == "unverified"
    record = report.local_descent.to_dict()["value"]
    assert record["mode"] == "independent_objective_reconstruction"
    assert record["witness_count"] == 1
    assert record["unresolved"] == 0
    assert record["numerical_witnesses"][0]["index"] == 0
    np.testing.assert_array_equal(candidate, original)
    assert not hasattr(mop, "_gt_reference")


def test_default_and_aligned_options_retained():
    mop = mb.mops.DTLZ5(M=4)
    sample = np.array([_original_row()])
    default = mop.assess_gt(reference=sample, probes=16)
    assert default.local_descent.status == "not_applicable"
    with pytest.raises(ValueError, match="requires explicit aligned"):
        mop.assess_gt(reference=sample, probes=16, local_descent=True)
    with pytest.raises(ValueError, match="either aligned"):
        mop.assess_gt(
            reference=sample, probes=16, local_descent=True,
            objective_only_local_descent=True,
        )
    with pytest.raises(ValueError, match="do not supply decision_reference"):
        mop.assess_gt(
            reference=sample, probes=16,
            decision_reference=np.ones((1, mop.N))*0.5,
            objective_only_local_descent=True,
        )


def test_unverified_not_promoted_when_no_witness():
    mop = mb.mops.DTLZ6(M=4)
    y = np.array([_original_row(g=0.0)])
    a = mop.assess_gt(
        reference=y, probes=16, objective_only_local_descent=True
    )
    assert a.local_descent.status == "unverified"
    assert a.local_descent.value["witness_count"] == 0
    assert a.local_descent.value["unresolved"] == 1


def test_other_mops_reject_opt_in():
    mop = mb.mops.DTLZ2(M=4)
    with pytest.raises(ValueError, match="only original DTLZ5/6"):
        mop.assess_gt(
            reference=mop.pf(n_points=12), probes=16,
            objective_only_local_descent=True,
        )


def test_report_uses_same_audit_without_changing_sidecar(tmp_path):
    mop = mb.mops.DTLZ5(M=4)
    path = tmp_path / "gt.json"
    y = np.array([_original_row()])
    assert mop.calibrate(source_baseline=path, source_gt=y)
    before = path.read_bytes()
    md = mop.report(
        show=False, markdown=True, probes=16,
        objective_only_local_descent=True,
    )
    assert "independent_objective_reconstruction" in md
    assert "local_descent" in md
    np.testing.assert_array_equal(mop.calibrated_gt, y)
    assert path.read_bytes() == before
