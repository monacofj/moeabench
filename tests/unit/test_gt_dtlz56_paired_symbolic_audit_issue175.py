# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive per-row conditional symbolic proof tests for original DTLZ5/6."""
from fractions import Fraction as Q

import numpy as np
import pytest

from scripts.audit.dtlz56_paired_gt_evidence import (
    _original_objectives, save_paired_candidate,
)
from scripts.audit.dtlz56_paired_symbolic_audit import (
    _exact_radial_g, _exact_tenth_root, audit_symbolic_candidate,
    audit_paired_bundle, main,
)


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_exact_decision_classification_separates_proven_and_unknown(problem):
    m,k=4,1
    x=np.zeros((6,m+k-1),dtype=float)
    x[:,:m-1]=[
        [.3,.3,.3],       # g0; global norm theorem
        [0,1,0],          # positive g, global supporting face
        [.25,.5,.5],      # positive g, same-ray strict dominator
        [.0625,1,0],      # positive g, nonzero first angle global support
        [0,1,1],          # positive g, all-upper objective support
        [.5,1,0],        # positive g, no applicable theorem
    ]
    x[:,m-1]=0.5 if problem=="DTLZ5" else 0.
    x[1:,m-1]=1.0
    f,_=_original_objectives(x,problem=problem,m=m,k=k)
    result=audit_symbolic_candidate(x,f,problem=problem,m=m,k=k)
    statuses=[entry["status"] for entry in result["rows"]]
    assert statuses==[
        "symbolic_decision_global_pf",
        "symbolic_decision_global_pf",
        "symbolic_decision_dominated",
        "symbolic_decision_global_pf",
        "symbolic_decision_global_pf",
        "unverified",
    ]
    assert result["counts"]=={
        "symbolic_decision_global_pf":4,
        "symbolic_decision_dominated":1,
        "unverified":1,
    }
    assert result["scientific_certification"]=="unverified"
    assert result["whole_reference_membership"]=="unverified"


def test_dtlz6_exact_tenth_root_requires_perfect_power_rationals():
    assert _exact_tenth_root(Q(0))==Q(0)
    assert _exact_tenth_root(Q(1,1024))==Q(1,2)
    assert _exact_tenth_root(Q(1))==Q(1)
    assert _exact_tenth_root(Q(1,2)) is None
    assert _exact_tenth_root(Q(1,3)) is None
    with pytest.raises(ValueError):
        _exact_tenth_root(Q(-1,2))


def test_dtlz5_radial_g_is_exact_binary_rational():
    decision=np.array([0.,1.,0.,.25,.75])
    expected=(Q(1,4)-Q(1,2))**2+(Q(3,4)-Q(1,2))**2
    assert _exact_radial_g(decision,problem="DTLZ5",m=4)==expected


def test_dtlz6_unknown_radial_value_cannot_be_promoted():
    m,k=4,1
    x=np.array([[0.,1.,0.,.5]])
    f,_=_original_objectives(x,problem="DTLZ6",m=m,k=k)
    a=audit_symbolic_candidate(x,f,problem="DTLZ6",m=m,k=k)
    assert a["rows"][0]["status"]=="unverified"
    assert a["rows"][0]["g_exact"] is None


def test_dtlz6_exact_nontrivial_root_can_prove_boundary():
    m,k=4,1
    x=np.array([[0.,1.,0.,1/1024]])
    f,_=_original_objectives(x,problem="DTLZ6",m=m,k=k)
    a=audit_symbolic_candidate(x,f,problem="DTLZ6",m=m,k=k)
    assert a["rows"][0]["status"]=="symbolic_decision_global_pf"
    assert a["rows"][0]["g_exact"]=={
        "numerator_hex":"0x1","denominator_hex":"0x2",
    }


def test_numerical_row_mismatch_blocks_symbolic_claim():
    m,k=4,1
    x=np.array([[0.,1.,0.,1.]])
    f,_=_original_objectives(x,problem="DTLZ6",m=m,k=k)
    f[0,1]+=0.01
    a=audit_symbolic_candidate(
        x,f,problem="DTLZ6",m=m,k=k,tolerance=1e-10,
    )
    assert a["independent_objective_alignment"]=="fail"
    assert a["rows"][0]["status"]=="unverified"


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_real_paired_artifact_replay_never_claims_full_gt(problem,tmp_path,capsys):
    m,k=4,1
    x=np.array([[0.,1.,0.,.5 if problem=="DTLZ5" else 0.],
                [0.,1.,0.,1.]])
    f,_=_original_objectives(x,problem=problem,m=m,k=k)
    artifact=tmp_path/"paired.npz"
    saved=save_paired_candidate(
        artifact,x,f,problem=problem,m=m,k=k,
    )
    report=audit_paired_bundle(
        artifact,problem=problem,m=m,k=k,
        expected_sha256=saved["artifact_sha256"],
    )
    assert report["counts"]["symbolic_decision_global_pf"]==2
    assert report["artifact_sha256"]==saved["artifact_sha256"]
    assert main(["--bundle",str(artifact),"--problem",problem,
                 "--m","4","--k","1",
                 "--expected-sha256",saved["artifact_sha256"]])==0
    assert "scientific_certification" in capsys.readouterr().out
    with pytest.raises(ValueError,match="digest mismatch"):
        audit_paired_bundle(
            artifact,problem=problem,m=m,k=k,expected_sha256="0"*64,
        )
    with pytest.raises(ValueError,match="metadata"):
        audit_paired_bundle(artifact,problem=problem,m=m,k=2)


def test_per_row_audit_does_not_sample_or_write(tmp_path,monkeypatch):
    from moeabench.mops import DTLZ5
    def disallowed(*args,**kwargs):
        raise AssertionError("No sampler/evaluator should be invoked")
    monkeypatch.setattr(DTLZ5,"ps",disallowed)
    monkeypatch.setattr(DTLZ5,"evaluation",disallowed)
    x=np.array([[0.,1.,0.,.5]])
    f,_=_original_objectives(x,problem="DTLZ5",m=4,k=1)
    before=set(tmp_path.iterdir())
    r=audit_symbolic_candidate(x,f,problem="DTLZ5",m=4,k=1)
    assert r["counts"]["symbolic_decision_global_pf"]==1
    assert set(tmp_path.iterdir())==before
