# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DTLZ5/6 PS-decision witness evidence (issue #175)."""
import json
import numpy as np
import pytest

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_decision_witness import (
    inspect_dtlz56_decision_witness,
)


def _published_witness(klass, m=4, g=.2, k=10):
    n = m+k-1
    x = np.empty((2,n), dtype=float)
    x[:, 0] = [0.2, 0.7]
    x[:, 1:m-1] = [[.5]*(m-2), [0., 1.]][:2]
    if klass is DTLZ5:
        x[0, m-1:] = .5
        x[1, m-1:] = .5+np.sqrt(g/k)
    else:
        x[0, m-1:] = 0.
        x[1, m-1:] = (g/k)**10
    return x


@pytest.mark.parametrize("klass", [DTLZ5, DTLZ6])
def test_original_decision_witness_matches_declared_objectives(klass):
    mop = klass(M=4)
    x = _published_witness(klass)
    # Use original evaluator only to construct fixture. The independent
    # oracle below must not invoke it and can detect a forged F.
    f = mop.evaluation(x)["F"].copy()
    def forbidden(*args, **kwargs):
        pytest.fail("independent witness check called production evaluator")
    mop.evaluation = forbidden
    observed = mop.assess_gt(
        reference=f, decision_reference=x, probes=16)
    assert observed.feasibility.status == "pass", observed.feasibility.to_dict()
    assert observed.feasibility.value["violations"] == 0
    assert observed.feasibility.value["max_objective_residual"] < 1e-10
    # Instance monkeypatch means no published-global oracle is inherited.
    assert observed.validity.status == "unverified"
    json.dumps(observed.to_dict(), allow_nan=False)


@pytest.mark.parametrize("klass", [DTLZ5, DTLZ6])
def test_mismatched_declared_objectives_fail_independent_witness(klass):
    mop = klass(M=4)
    x = _published_witness(klass)
    f = mop.evaluation(x)["F"].copy()
    f[-1, 2] += .1
    result = mop.assess_gt(
        reference=f, decision_reference=x, probes=16)
    assert result.feasibility.status == "fail"
    assert result.feasibility.value["objective_mismatch"] == 1


@pytest.mark.parametrize("klass", [DTLZ5, DTLZ6])
def test_out_of_bounds_decision_is_rejected_without_evaluator(klass):
    mop = klass(M=4)
    x = _published_witness(klass)
    f = mop.evaluation(x)["F"].copy()
    x[0, 0] = 1.25
    assessment = mop.assess_gt(
        reference=f, decision_reference=x, probes=16)
    assert assessment.feasibility.status == "fail"
    assert assessment.feasibility.value["invalid_decisions"] == 1


def test_decision_reference_row_alignment_is_required():
    mop = DTLZ5(M=4)
    x = _published_witness(DTLZ5)
    f = mop.evaluation(x)["F"]
    with pytest.raises(ValueError, match="aligned rows"):
        mop.assess_gt(reference=f, decision_reference=x[:1], probes=16)
    with pytest.raises(ValueError, match="explicit objective reference"):
        mop.assess_gt(n_points=2, decision_reference=x, probes=16)


@pytest.mark.parametrize("klass", [DTLZ5, DTLZ6])
def test_uncertified_objectives_without_witness_remain_unverified(klass):
    mop = klass(M=4)
    x = _published_witness(klass)
    f = mop.evaluation(x)["F"]
    result = mop.assess_gt(reference=f, probes=16)
    assert result.feasibility.status == "unverified"


def test_report_verifies_supplied_decision_witness_without_sidecar_mutation(tmp_path):
    mop = DTLZ5(M=3)
    x = mop.ps(n_points=5)
    f = mop.evaluation(x)["F"]
    source = tmp_path/"gt.json"
    mop.calibrate(source_gt=f, source_baseline=source)
    before = source.read_bytes()
    report = mop.report(
        show=False, markdown=False, decision_reference=x, probes=16)
    assert "feasibility" in report
    assert "independent published DTLZ5/6" in report
    assert source.read_bytes() == before


def test_other_mops_do_not_inherit_original_dtlz56_oracle():
    from moeabench.mops import DTLZ2
    mop = DTLZ2(M=3)
    x = mop.ps(n_points=7)
    f = mop.evaluation(x)["F"]
    result = mop.assess_gt(reference=f, decision_reference=x, probes=16)
    assert result.feasibility.status == "unverified"
