# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive paired GT evidence tests; never modify baseline or regression."""
import json

import numpy as np
import pytest

from scripts.audit.dtlz56_paired_gt_evidence import (
    _original_objectives, audit_paired_candidate,
    save_paired_candidate, verify_paired_candidate,
    main,
)


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [2,3,4,5,8])
def test_independent_published_image_and_aligned_pair(problem,m):
    k=4
    rng=np.random.default_rng(m+412)
    x=rng.uniform(0,1,size=(6,m+k-1))
    x[0,m-1:]=0.5 if problem=="DTLZ5" else 0
    x[1,0]=0
    x[1,1:m-1]=0.5
    y,g=_original_objectives(x,problem=problem,m=m,k=k)
    assert y.shape==(6,m)
    assert np.allclose(np.linalg.norm(y,axis=1),1+g,rtol=0,atol=1e-11)
    result=audit_paired_candidate(x,y,problem=problem,m=m,k=k)
    assert result["independent_objective_alignment"]=="pass"
    assert result["decision_feasibility"]=="pass"
    assert result["zero_g_rows"]>=1
    assert result["global_pareto_membership"]=="unverified"
    assert result["scientific_certification"]=="unverified"


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_candidate_saved_and_replayed_with_optional_sidecar(problem,tmp_path):
    m,k=4,4
    x=np.zeros((4,m+k-1))
    x[:,0]=[0,.25,.5,1]
    x[:,1:m-1]=[[1,0],[1,1],[.5,.5],[.5,0]]
    x[:,m-1:]=.5 if problem=="DTLZ5" else 0
    y,_=_original_objectives(x,problem=problem,m=m,k=k)
    file=tmp_path/"paired.npz"
    record=save_paired_candidate(
        file,x,y,problem=problem,m=m,k=k)
    assert record["independent_objective_alignment"]=="pass"
    assert file.is_file()
    with pytest.raises(FileExistsError):
        save_paired_candidate(
            file,x,y,problem=problem,m=m,k=k)
    sidecar=tmp_path/"sidecar.json"
    sidecar.write_text(json.dumps({
        "problem_id":problem,"mop_dimension":m,
        "gt_source":"analytical","gt_reference":y.tolist()
    }),encoding="utf-8")
    verified=verify_paired_candidate(
        file,problem=problem,m=m,k=k,sidecar=sidecar,
        expected_sha256=record["artifact_sha256"])
    assert verified["expected_digest_match"] is True
    assert verified["sidecar_alignment"]=="numerically_aligned"
    assert verified["scientific_certification"]=="unverified"
    assert main(["verify","--bundle",str(file),"--problem",problem,
                 "--m",str(m),"--k",str(k),"--sidecar",str(sidecar),
                 "--expected-sha256",record["artifact_sha256"]])==0


def test_modified_objective_row_detected_before_persist(tmp_path):
    m,k=4,4
    x=np.full((3,m+k-1),.5)
    y,_=_original_objectives(x,problem="DTLZ5",m=m,k=k)
    y[1,2]+=0.01
    outcome=audit_paired_candidate(
        x,y,problem="DTLZ5",m=m,k=k,tolerance=1e-10)
    assert outcome["independent_objective_alignment"]=="fail"
    assert outcome["misaligned_rows"]==1
    path=tmp_path/"wrong.npz"
    with pytest.raises(ValueError,match="misaligned"):
        save_paired_candidate(
            path,x,y,problem="DTLZ5",m=m,k=k,tolerance=1e-10)
    assert not path.exists()


def test_bad_sidecar_order_or_metadata_not_accepted(tmp_path):
    m,k=4,4
    x=np.full((3,m+k-1),.5)
    x[0,0]=0
    x[1,0]=.3
    x[2,0]=.8
    y,_=_original_objectives(x,problem="DTLZ5",m=m,k=k)
    sidecar=tmp_path/"prior.json"
    sidecar.write_text(json.dumps({
        "problem_id":"DTLZ5", "mop_dimension":m,
        "gt_reference":y[::-1].tolist()
    }),encoding="utf-8")
    result=audit_paired_candidate(
        x,y,problem="DTLZ5",m=m,k=k,sidecar=sidecar)
    assert result["independent_objective_alignment"]=="pass"
    assert result["sidecar_alignment"]=="mismatched_objective_rows"
    sidecar.write_text(json.dumps({
        "problem_id":"DTLZ6", "mop_dimension":m,
        "gt_reference":y.tolist()
    }),encoding="utf-8")
    assert audit_paired_candidate(
        x,y,problem="DTLZ5",m=m,k=k,sidecar=sidecar
    )["sidecar_alignment"]=="mismatched_metadata"


def test_tampered_bundle_and_wrong_digest_cannot_pass(tmp_path):
    m,k=4,4
    x=np.full((3,m+k-1),.5)
    y,_=_original_objectives(x,problem="DTLZ5",m=m,k=k)
    path=tmp_path/"bundle.npz"
    record=save_paired_candidate(path,x,y,problem="DTLZ5",m=m,k=k)
    bad=verify_paired_candidate(
        path,problem="DTLZ5",m=m,k=k,expected_sha256="0"*64)
    assert bad["expected_digest_match"] is False
    assert main(["verify","--problem","DTLZ5","--m","4","--k","4",
                 "--bundle",str(path),"--expected-sha256","0"*64])==1
    with path.open("wb") as f:
        np.savez_compressed(f,decisions=x,objectives=y+1,
                            problem="DTLZ5",M=m,K=k)
    bad=verify_paired_candidate(path,problem="DTLZ5",m=m,k=k)
    assert bad["independent_objective_alignment"]=="fail"
    assert bad["artifact_sha256"]!=record["artifact_sha256"]


@pytest.mark.parametrize("invalid",[
    [[-0.1]*7],[[float("nan")]*7],[[1.01]*7],[[.5]*6]
])
def test_infeasible_or_malformed_decisions_rejected(invalid):
    with pytest.raises(ValueError):
        _original_objectives(
            invalid,problem="DTLZ5",m=4,k=4)


def test_no_implicit_heavy_generation_and_validation(tmp_path,monkeypatch):
    import scripts.audit.dtlz56_paired_gt_evidence as p
    def forbidden(*args,**kwargs):
        raise AssertionError("Sampler must not run on check")
    import moeabench.mops as mops
    monkeypatch.setattr(mops.DTLZ5,"ps",forbidden)
    x=np.full((2,7),.5)
    y,_=p._original_objectives(x,problem="DTLZ5",m=4,k=4)
    record=p.audit_paired_candidate(x,y,problem="DTLZ5",m=4,k=4)
    assert record["independent_objective_alignment"]=="pass"
    # The standalone artifact can be created without calling .ps().
    p.save_paired_candidate(tmp_path/"manual.npz",x,y,
                            problem="DTLZ5",m=4,k=4)
