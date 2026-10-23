#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reproducible NONCERTIFIED FAIR #75 candidate baseline manifest.

A manifest is not an authorization to install a baseline. It binds immutable
SHA256 hashes, the *exact* normalizing coordinates, area sampling method and
seed, targeted K records, independent verification, and unsupported scope.
Never rewrites installed reference packages or any official audit report.
"""
import argparse
import hashlib
import json
from pathlib import Path

from calibration.scripts.verify_closeness_migration import verify

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = (ROOT/"moeabench/diagnostics/resources").resolve()
REPORTS = (ROOT/"calibration/reports").resolve()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_manifest(original,candidate,impact,verified,*,sha256,source_revision):
    if impact.get("status") != "NONCERTIFIED_CANDIDATE":
        raise ValueError("Impact lacks NONCERTIFIED_CANDIDATE status")
    if impact.get("reference_law") != "normalized_objective_area":
        raise ValueError("Candidate uses a different probability measure")
    targeted=tuple(sorted(impact.get("problems",{})))
    if targeted!=("DTLZ1","DTLZ2") or impact.get("failed"):
        raise ValueError("Only independently justified DTLZ1/2 area-law coverage allowed")
    check=verify(original,candidate,list(targeted))
    if verified.get("status") != check["status"] or (
        verified.get("k_counts") != check["k_counts"] or
        verified.get("untouched_problem_count") != check["untouched_problem_count"]
    ):
        raise ValueError("Independent verification provenance does not match candidate")
    problems={}
    for name in sorted(original["problems"]):
        if name not in targeted:
            problems[name]={
                "status":"UNSUPPORTED_FOR_THIS_OBJECTIVE_AREA_CANDIDATE",
                "reason":"No independently validated intrinsic-area sampling law in this candidate",
            }
            continue
        rec=impact["problems"][name]
        provenance=rec.get("ideal_sample_provenance")
        if (not isinstance(provenance,dict) or
            provenance.get("target_measure")!="uniform_intrinsic_area_normalized_objective_pf" or
            provenance.get("n_fresh",0)<2 or
            not isinstance(provenance.get("seed"),int)):
            raise ValueError("Missing accepted pilot provenance: "+name)
        if set(rec.get("K",{}))!=set(check["k_records"][name]):
            raise ValueError("Impact and independently verified K coverage differ")
        for k in rec["K"]:
            if rec["K"][k]["new_ideal_res"]!=check["k_records"][name][k]["epsilon"]:
                raise ValueError("Impact epsilon differs from independently verified baseline")
        problems[name]={
            "status":"PROVISIONAL_NOT_CERTIFIED",
            "objective_dimension":3,"intrinsic_pf_dimension":2,
            "reference_measure":provenance["target_measure"],
            "analytical_sampler":provenance["law"],
            "normalization":{"ideal":provenance["ideal"],
                             "nadir":provenance["nadir"]},
            "gt_sha256":rec["gt_sha256"],
            "gt_points":rec["gt_points"],
            "ideal_sampling_seed":provenance["seed"],
            "ideal_sample_count":provenance["n_fresh"],
            "K_count":len(rec["K"]),
            "K_values":sorted(int(k) for k in rec["K"]),
            "max_ideal_res":max(x["new_ideal_res"] for x in rec["K"].values()),
            "min_ideal_res":min(x["new_ideal_res"] for x in rec["K"].values()),
        }
    if len(problems)!=14:
        raise ValueError("Expected complete frozen 14-MOP inventory")
    return {
        "id":"FAIR_ISSUE75_NORMALIZED_AREA_M3_CANDIDATE_V1",
        "status":"NONCERTIFIED_EXPERIMENTAL_MANIFEST",
        "for_publication":False,
        "replaces_certified_baseline":False,
        "source_revision":source_revision,
        "source_baseline_sha256":sha256["original"],
        "candidate_baseline_sha256":sha256["candidate"],
        "impact_report_sha256":sha256["impact"],
        "independent_verification_sha256":sha256["verified"],
        "source_version":original.get("version"),
        "candidate_method":"uniform_normalized_objective_pf_area_v1",
        "K_records_provisionally_checked":sum(check["k_counts"].values()),
        "problems":problems,
        "other_metrics_unchanged":True,
        "historical_headway_policy":"REQUIRE_TRUE_GENERATION_ZERO_ELSE_UNAVAILABLE",
        "notes":[
            "Uniform area is a proposed normative choice, not an accepted FAIR release policy",
            "Unsupported means unsupported for this area-law candidate, not unsupported FAIR generally",
            "Versioned source baseline and frozen GT are unchanged",
            "A committed manifest alone does not certify Q or regenerate an audit_report",
        ],
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--impact",type=Path,required=True)
    p.add_argument("--verified",type=Path,required=True)
    p.add_argument("--source-revision",default="unknown")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=args.output.resolve()
    if out.exists() or out.is_relative_to(PROTECTED) or out.is_relative_to(REPORTS) or (
        out==(ROOT/"calibration/audit_report.html").resolve()):
        p.error("Refusing overwrite or installed/certified destination")
    inputs=[args.original,args.candidate,args.impact,args.verified]
    data=[json.loads(x.read_text()) for x in inputs]
    manifest=build_manifest(*data,sha256=dict(zip(
        ("original","candidate","impact","verified"),
        (digest(x) for x in inputs))),
        source_revision=args.source_revision)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "manifest":manifest["id"],
        "status":manifest["status"],
        "K_checked":manifest["K_records_provisionally_checked"],
        "covered":["DTLZ1","DTLZ2"],
        "not_area_certified":len(manifest["problems"])-2,
        "source_revision":manifest["source_revision"],
    },indent=2))
if __name__=="__main__":
    main()
