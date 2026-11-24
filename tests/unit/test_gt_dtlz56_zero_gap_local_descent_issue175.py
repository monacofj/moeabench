# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Zero-gap local dominance theorem checks, independent of production PF/PS."""
import json
import numpy as np
import pytest

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_local_descent import (
    audit_local_dtlz56_descent, _objective_and_derivative,
)


def _decision(problem, u=(.25,.7,0.), g=.2, k=10):
    m=len(u)+1
    x=np.empty(m+k-1)
    x[:m-1]=u
    x[m-1:] = (
        .5+np.sqrt(g/k) if problem=="DTLZ5"
        else (g/k)**10)
    return x


@pytest.mark.parametrize("problem,klass", [("DTLZ5",DTLZ5),("DTLZ6",DTLZ6)])
def test_strict_first_order_direction_constructs_real_positive_g_dominator(
        problem,klass,monkeypatch):
    x=_decision(problem)
    candidate=klass(M=4).evaluation(x)["F"][0]
    outcome=audit_local_dtlz56_descent(x,problem=problem,m=4)
    assert outcome.status=="witness",outcome.to_dict()
    assert outcome.derivative_margin>0
    assert outcome.witness_g<.2
    witness=np.array(outcome.witness_objectives)
    assert np.all(witness<candidate-1e-10)
    assert outcome.min_objective_improvement>1e-10
    w=np.array(outcome.witness_decision)
    assert np.all((0<=w)&(w<=1))
    assert len(w)==len(x)
    json.dumps(outcome.to_dict(),allow_nan=False)

    def forbidden(*args,**kwargs):
        pytest.fail("production evaluator reused inside independent local oracle")
    monkeypatch.setattr(klass,"evaluation",forbidden)
    second=audit_local_dtlz56_descent(x,problem=problem,m=4)
    assert second.status=="witness"
    np.testing.assert_array_less(witness,candidate)
    f=outcome.to_dict()
    assert f["status"]=="witness"


@pytest.mark.parametrize("problem", ["DTLZ5","DTLZ6"])
def test_derivative_prediction_matches_one_sided_finite_difference(problem):
    x=_decision(problem,u=(.31,.7,.2),g=.2)
    g=.2
    u=x[:3]
    f,b,a=_objective_and_derivative(u,g)
    v=np.array([.02,-.03,.05])
    eps=1e-6
    lower,_,_=_objective_and_derivative(u-eps*v,g-eps)
    np.testing.assert_allclose((f-lower)/eps,b+a@v,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize("problem", ["DTLZ5","DTLZ6"])
def test_boundary_direction_without_strict_margin_is_not_falsely_certified(problem):
    x=_decision(problem,u=(.25,1.,0.),g=.2)
    out=audit_local_dtlz56_descent(x,problem=problem,m=4)
    assert out.status in ("unverified","witness")
    # 'unverified' from first-order search cannot be called global optimality.


def test_opt_in_assessment_and_report_remain_read_only(tmp_path):
    mop=DTLZ5(M=4)
    x=_decision("DTLZ5")
    f=mop.evaluation(x)["F"]
    a=mop.assess_gt(reference=f,decision_reference=x[None,:],
                    local_descent=True,probes=16)
    assert a.local_descent.status=="unverified"
    assert a.local_descent.value["witness_count"]==1
    assert a.local_descent.value["numerical_witnesses"][0]["index"]==0
    assert a.validity.status=="unverified"  # whole-front certificate remains withheld
    json.dumps(a.to_dict(),allow_nan=False)
    ordinary=mop.assess_gt(reference=f,probes=16)
    assert ordinary.local_descent.status=="not_applicable"

    sidecar=tmp_path/"gt.json"
    mop.calibrate(source_gt=f,source_baseline=sidecar)
    before=sidecar.read_bytes()
    output=mop.report(show=False,markdown=False,decision_reference=x[None,:],
                      local_descent=True,probes=16)
    assert "local_descent" in output
    assert "witness_count" in output
    assert sidecar.read_bytes()==before


def test_must_explicitly_supply_aligned_decisions():
    mop=DTLZ5(M=4)
    f=mop.evaluation(_decision("DTLZ5"))["F"]
    with pytest.raises(ValueError,match="requires explicit aligned"):
        mop.assess_gt(reference=f,local_descent=True,probes=16)
    with pytest.raises(ValueError,match="only original DTLZ5/6"):
        from moeabench.mops import DTLZ2
        x=DTLZ2(M=4).ps(n_points=3)
        DTLZ2(M=4).assess_gt(
            reference=DTLZ2(M=4).evaluation(x)["F"],
            decision_reference=x,local_descent=True,probes=16)


@pytest.mark.parametrize("bad", [
    {"numerical_tolerance":0},
    {"max_halvings":0},
    {"numerical_tolerance":np.nan},
])
def test_invalid_local_protocol_is_rejected(bad):
    with pytest.raises(ValueError):
        audit_local_dtlz56_descent(
            _decision("DTLZ5"),problem="DTLZ5",m=4,**bad)


def test_forged_objective_reference_never_inherits_local_descent_witness():
    mop=DTLZ5(M=4)
    x=_decision("DTLZ5")
    f=mop.evaluation(x)["F"].copy()
    f[0,0]+=.2
    out=mop.assess_gt(
        reference=f,decision_reference=x[None,:],
        local_descent=True,probes=16)
    assert out.feasibility.status=="fail"
    assert out.local_descent.value["witness_count"]==0
    assert out.local_descent.value["unresolved"]==1
