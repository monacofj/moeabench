# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #75: prospective manifest and report never impersonate certification."""
import copy
import pytest
from calibration.scripts.build_fair_candidate_manifest_issue75 import build_manifest
from calibration.scripts.render_fair_release_preview_issue75 import render_versioned_preview

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

def test_manifest_records_supported_scope_and_all_other_mops_explicit():
    m=_manifest()
    assert m["status"]=="NONCERTIFIED_EXPERIMENTAL_MANIFEST"
    assert m["K_records_provisionally_checked"]==2
    assert m["source_revision"]=="frozen-commit"
    assert len(m["problems"])==14
    assert m["problems"]["DTLZ1"]["status"]=="PROVISIONAL_NOT_CERTIFIED"
    assert m["problems"]["DPF5"]["status"]=="UNSUPPORTED_FOR_THIS_OBJECTIVE_AREA_CANDIDATE"

def test_manifest_rejects_tampered_verification_and_reference_law():
    a,b,c,d,h=_inputs()
    d["k_counts"]["DTLZ1"]=2
    with pytest.raises(ValueError,match="Independent verification"):
        build_manifest(a,b,c,d,sha256=h,source_revision="a")
    d["k_counts"]["DTLZ1"]=1
    c["reference_law"]="decision_uniform"
    with pytest.raises(ValueError,match="probability measure"):
        build_manifest(a,b,c,d,sha256=h,source_revision="a")

def test_versioned_report_explicitly_withholds_historical_q_and_real_p0_provenance():
    body=render_versioned_preview(_manifest(),_q(),_history())
    assert "NOT CERTIFIED" in body
    assert "985 recorded runs" in body
    assert "0 actual generation-zero" in body
    assert "Historic Q-scores are" in body
    assert "No synthetic generation-zero" in body
    assert "truep0sha" in body  # Audit links genuine P0 hash to its optimizer seed.
    assert "0.4800000" in body
    assert "UNSUPPORTED_FOR_THIS_OBJECTIVE_AREA_CANDIDATE" in body

def test_versioned_report_rejects_hash_mismatch_and_fabricated_p0():
    m=_manifest();q=_q();h=_history()
    q["baseline_candidate_sha256"]="different"
    with pytest.raises(ValueError,match="hashes"):
        render_versioned_preview(m,q,h)
    q["baseline_candidate_sha256"]="cand"
    q["trials"][0]["P0_sha256"]=""
    with pytest.raises(ValueError,match="real P0"):
        render_versioned_preview(m,q,h)
    q["trials"][0]["P0_sha256"]="p0"
    h["report_generation_safe"]=True
    with pytest.raises(ValueError,match="certification"):
        render_versioned_preview(m,q,h)

def test_versioned_report_escapes_untrusted_source_revision():
    m=_manifest()
    m["source_revision"]="<script>alert(42)</script>"
    body=render_versioned_preview(m,_q(),_history())
    assert "<script" not in body
    assert "&lt;script&gt;" in body


def test_versioned_report_shows_release_blockers_and_cannot_claim_certification(monkeypatch):
    from calibration.scripts.assess_fair_release_issue75 import assess_release
    monkeypatch.setattr("moeabench.system.version",lambda show=False:"0.16.1")
    manifest=_manifest()
    manifest["source_version"]="0.14.0"
    env={
        "library_version":"0.16.1",
        "version_check":{"status":"MISMATCH","differences":{}},
        "stability_check":{"status":"PASSED","checks":{"prng":"PASS"}},
        "runtime":{},
    }
    assessment=assess_release(manifest,_q(),_history(),env)
    body=render_versioned_preview(manifest,_q(),_history(),assessment)
    assert "PROPOSED AREA-LAW CALIBRATION NOT CERTIFIED" in body
    assert "ARCHIVED_SOURCE_VERSION_DIFFERS_FROM_CONSUMER" in body
    assert "source/consumer version difference is normal" in body
    assert "Scientific decision blockers" in body
    assert "Version/dependency provenance notes (not blockers)" in body
    assert "0.14.0" in body and "0.16.1" in body
    assert "PASSED" in body
    assert "No synthetic generation-zero populations" in body
    assessment["calibration_provenance"]["candidate_baseline_sha256"]="spoofed"
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        render_versioned_preview(manifest,_q(),_history(),assessment)
