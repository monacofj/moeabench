# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #75: fail-closed compatibility release gate cannot retag provenance."""
import copy
import pytest

from calibration.scripts.assess_fair_release_issue75 import assess_release
from calibration.scripts.build_fair_candidate_manifest_issue75 import build_manifest

MOPS=("DPF1","DPF2","DPF3","DPF4","DPF5",
      "DTLZ1","DTLZ2","DTLZ3","DTLZ4","DTLZ5",
      "DTLZ6","DTLZ7","DTLZ8","DTLZ9")

def _inputs():
    original={"version":"0.14.0","problems":{}}
    candidate=copy.deepcopy(original)
    for name in MOPS:
        clos={"rand_ecdf":[.2,.4,.7],"rand50":.4,
              "p95":.7,"ideal_res":0.0}
        record={"closeness":clos,"cov":{"rand50":.2},
                "headway":{"rand50":.5}}
        original["problems"][name]={"20":copy.deepcopy(record),
                                    "gt_reference":[[.5,.5,.0]]}
        candidate["problems"][name]=copy.deepcopy(original["problems"][name])
    impact={"status":"NONCERTIFIED_CANDIDATE",
            "reference_law":"normalized_objective_area",
            "failed":{},"problems":{}}
    for name,sampler in (
        ("DTLZ1","uniform_simplex_area_dirichlet_111"),
        ("DTLZ2","uniform_normalized_dtlz2_surface_area_rejection"),
    ):
        clos=candidate["problems"][name]["20"]["closeness"]
        clos.update({"rand_ecdf":[.1,.3,.6],"rand50":.3,
                     "p95":.6,"ideal_res":.1,
                     "ideal_res_method":"uniform_normalized_objective_pf_area_v1",
                     "reference_measure":"uniform_intrinsic_area_normalized_objective_pf",
                     "ideal_res_sampler":sampler,"ideal_res_sampling_seed":123,
                     "ideal_res_sample_count":4096})
        impact["problems"][name]={
            "gt_sha256":name+"gtsha","gt_points":10000,
            "ideal_sample_provenance":{
                "target_measure":"uniform_intrinsic_area_normalized_objective_pf",
                "law":sampler,"ideal":[0.,0.,0.],
                "nadir":[1.,1.,1.],"seed":123,"n_fresh":4096,
            },
            "K":{"20":{"new_ideal_res":.1}},
        }
    verification={"status":"PASS_PROVISIONAL_CLOSENESS_ONLY",
                  "k_counts":{"DTLZ1":1,"DTLZ2":1},
                  "untouched_problem_count":12}
    digests={"original":"froz","candidate":"cand",
             "impact":"impacthash","verified":"verifyhash"}
    return original,candidate,impact,verification,digests

def _manifest():
    a,b,c,d,h=_inputs()
    return build_manifest(a,b,c,d,sha256=h,source_revision="frozen-commit")

def _q():
    return {
        "status":"NONCERTIFIED_PAIRED_REAL_MOEA_Q_CANDIDATE",
        "certified":False,
        "problem":"DTLZ1",
        "baseline_candidate_sha256":"cand",
        "baseline_frozen_sha256":"froz",
        "reference_measure":"uniform_intrinsic_area_normalized_objective_pf",
        "sample_seed":123,"sample_count":4096,
        "frozen_gt_sha256":"DTLZ1gtsha",
        "unrelated_fair_q_unchanged":True,
        "K":100,"generations":500,"area_ideal_res":.04,
        "informative_checkpoint_count":1,"clipped_checkpoint_count":1,
        "all_checkpoint_count":2,
        "max_informative_absolute_q_shift":.08,
        "trials":[{"optimizer_seed":1,"P0_sha256":"truep0sha",
                   "P0_points":100,"headway_same_under_candidate":True,
                   "checkpoints":[{"generation":0,"front_points":60,
                         "q_frozen":0.0,"q_area_candidate":0.0,
                         "absolute_q_shift":0.0,"both_endpoints_clipped":True},
                        {"generation":-1,"front_points":100,
                         "q_frozen":.4,"q_area_candidate":.48,
                         "absolute_q_shift":.08,"both_endpoints_clipped":False}
                   ]}],
    }

def _history():
    return {
        "scope":"file-presence preflight only, not calibration certification",
        "report_generation_safe":False,
        "run_records":985,"final_front_files":985,
        "with_true_generation_zero":0,"headway_data_missing_runs":985,
    }




def _env(monkeypatch, version_status="MISMATCH", stability_status="PASSED"):
    monkeypatch.setattr("moeabench.system.version", lambda show=False: "0.16.1")
    return {
        "library_version":"0.16.1",
        "version_check":{"status":version_status,
                         "differences":{"numpy":{"certified":"old","current":"new"}}},
        "stability_check":{"status":stability_status,
                           "checks":{"python_random":"PASS",
                                     "numpy_generator":"PASS"}},
        "runtime":{"python":"3.14.4","numpy":"2.5.4"},
    }


def test_separate_source_baseline_version_from_library_version(monkeypatch):
    env=_env(monkeypatch)
    m=_manifest()
    m["source_version"]="0.14.0"
    result=assess_release(m,_q(),_history(),env)
    assert result["status"]=="BLOCKED_PENDING_SCIENTIFIC_RELEASE_REVIEW"
    assert result["certified"] is False
    assert result["publishable"] is False
    assert result["overwrite_canonical_allowed"] is False
    assert result["calibration_provenance"]["original_baseline_version"]=="0.14.0"
    assert result["calibration_provenance"]["consumer_library_version"]=="0.16.1"
    assert result["checks"]["numerical_stability_smoke_passed"] is True
    assert result["paired_q_impact"]["allowable_drift_threshold"] is None
    blockers={x["code"] for x in result["release_blockers"]}
    assert "ARCHIVED_BASELINE_LIBRARY_VERSION_DIFFER" in blockers
    assert "DEPENDENCY_CERTIFICATION_MISSING" in blockers
    assert "BREAKING_Q_CHANGE_REQUIRES_REVIEW" in blockers
    assert "HISTORICAL_P0_UNOBSERVABLE" in blockers


def test_no_release_without_stability_even_when_versions_match(monkeypatch):
    env=_env(monkeypatch,version_status="MATCH",stability_status="FAILED")
    m=_manifest()
    m["source_version"]="0.16.1"
    result=assess_release(m,_q(),_history(),env)
    assert result["checks"]["source_calibration_and_runtime_version_equal"] is True
    assert result["checks"]["numerical_stability_smoke_passed"] is False
    assert "NUMERICAL_STABILITY_UNVERIFIED_OR_FAILED" in {
        x["code"] for x in result["release_blockers"]
    }
    assert result["publishable"] is False


def test_fail_closed_on_wrong_hash_measure_history_or_environment(monkeypatch):
    env=_env(monkeypatch)
    m=_manifest()
    m["source_version"]="0.14.0"
    q=_q()
    q["baseline_candidate_sha256"]="spoofed"
    with pytest.raises(ValueError,match="hashes"):
        assess_release(m,q,_history(),env)
    q=_q()
    q["informative_checkpoint_count"]=0
    with pytest.raises(ValueError,match="nonclipped"):
        assess_release(m,q,_history(),env)
    h=_history()
    h["report_generation_safe"]=True
    with pytest.raises(ValueError,match="Historical provenance"):
        assess_release(m,_q(),h,env)
    bad=copy.deepcopy(env)
    bad["stability_check"]["status"]="UNCHECKED"
    with pytest.raises(ValueError,match="actually measured"):
        assess_release(m,_q(),_history(),bad)
    bad=copy.deepcopy(env)
    bad["library_version"]="0.14.0"
    with pytest.raises(ValueError,match="not for this MoeaBench"):
        assess_release(m,_q(),_history(),bad)


def test_reject_mop_reclassification_without_scientific_provenance(monkeypatch):
    env=_env(monkeypatch)
    m=_manifest()
    m["source_version"]="0.14.0"
    m["problems"]["DPF5"]["status"]="PROVISIONAL_NOT_CERTIFIED"
    with pytest.raises(ValueError,match="scope"):
        assess_release(m,_q(),_history(),env)
