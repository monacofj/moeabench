#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #75: fail-closed scientific release/readiness assessment.

A frozen baseline's version is *its original calibration provenance*, not a
claim that the current library has the same version. A source/consumer version
difference by itself is NOT a reason to recalibrate or block publication. A reproducible paired Q
study, and even a passing numerical smoke/stability check, do not certify a
new probability law or retroactively validate archived Q values. This tool
joins evidence without rewriting that history or setting post-hoc tolerances.
"""
import argparse
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RESOURCES=(ROOT/"moeabench/diagnostics/resources").resolve()
REPORTS=(ROOT/"calibration/reports").resolve()
CANONICAL=(ROOT/"calibration/audit_report.html").resolve()


def assess_release(manifest, q_study, historical, environment):
    """Make unsupported and release-decision dependencies machine-readable."""
    if (manifest.get("status")!="NONCERTIFIED_EXPERIMENTAL_MANIFEST" or
        manifest.get("for_publication") is not False or
        manifest.get("replaces_certified_baseline") is not False):
        raise ValueError("A scientific candidate must explicitly remain noncertified")
    if (q_study.get("status")!="NONCERTIFIED_PAIRED_REAL_MOEA_Q_CANDIDATE" or
        q_study.get("certified") is not False):
        raise ValueError("Need actual paired, explicitly noncertified clinical-Q evidence")
    if (manifest.get("candidate_baseline_sha256")!=q_study.get("baseline_candidate_sha256") or
        manifest.get("source_baseline_sha256")!=q_study.get("baseline_frozen_sha256")):
        raise ValueError("Baseline candidate/source hashes disagree")
    if (q_study.get("informative_checkpoint_count",0)<1 or
        q_study.get("max_informative_absolute_q_shift") is None or
        not math.isfinite(q_study["max_informative_absolute_q_shift"]) or
        q_study["max_informative_absolute_q_shift"]<0 or
        q_study.get("unrelated_fair_q_unchanged") is not True):
        raise ValueError("No valid nonclipped Q impact and unchanged-metric evidence")
    if (historical.get("scope")!="file-presence preflight only, not calibration certification" or
        historical.get("report_generation_safe") is not False or
        historical.get("headway_data_missing_runs",0)<1):
        raise ValueError("Historical provenance does not justify a missing-Headway report")
    for key in ("version_check","stability_check"):
        if not isinstance(environment.get(key),dict):
            raise ValueError("Missing independent environment verification: "+key)
    version=environment["version_check"]
    stability=environment["stability_check"]
    if version.get("status") not in ("MATCH","MISMATCH","UNCONFIGURED"):
        raise ValueError("Unknown certified-environment comparison")
    if stability.get("status") not in ("PASSED","FAILED"):
        raise ValueError("Numerical stability was not actually measured")
    from moeabench.system import version as library_version
    current=library_version(show=False)
    if environment.get("library_version")!=current:
        raise ValueError("Environment record is not for this MoeaBench revision")
    source=manifest.get("source_version")
    if not isinstance(source,str) or not source:
        raise ValueError("No archived source calibration version")

    expected_mops={"DTLZ1","DTLZ2"}
    supported={k for k,v in manifest.get("problems",{}).items()
               if v.get("status")=="PROVISIONAL_NOT_CERTIFIED"}
    if supported != expected_mops or len(manifest["problems"]) != 14:
        raise ValueError("Unexpected area-law certification scope")
    if q_study.get("problem") not in supported:
        raise ValueError("Paired clinical study is for an unsupported problem")
    if q_study.get("reference_measure") != manifest["problems"][q_study["problem"]].get("reference_measure"):
        raise ValueError("Q experiment uses an incompatible PF probability measure")

    checks={
        "candidate_and_q_hashes_linked":True,
        "paired_real_q_informative":True,
        "other_fair_and_q_metrics_unchanged":True,
        "proposed_area_law_scope_exactly_DTLZ1_DTLZ2_M3":True,
        "numerical_stability_smoke_passed":stability["status"]=="PASSED",
        "certified_dependency_versions_match":version["status"]=="MATCH",
        "source_calibration_and_runtime_version_equal":source==current,
        "historical_has_complete_true_P0":historical.get("headway_data_missing_runs")==0,
    }
    blockers=[
        {"code":"NORMATIVE_MEASURE_NOT_APPROVED",
         "reason":"Uniform normalized-objective PF area is proposed, not yet adopted in the normative FAIR ADR."},
        {"code":"BREAKING_Q_CHANGE_REQUIRES_REVIEW",
         "reason":"The observed Q-Closeness shift requires explicit numerical/clinical interpretation; never infer a release tolerance from the pilot."},
        {"code":"PARTIAL_MOP_OBJECTIVE_AREA_SUPPORT",
         "reason":"Only DTLZ1/DTLZ2 M3 have justified candidate surface samplers, while twelve other MOPs retain older/unsupported candidate semantics."},
        {"code":"HISTORICAL_P0_UNOBSERVABLE",
         "reason":"Archive lacks paired true generation-zero fronts; cannot reproduce longitudinal HEADWAY or its historical Q."},
        {"code":"CALIBRATION_FORMAT_AND_REPORT_RELEASE_REVIEW",
         "reason":"Frozen baseline and canonical audit-report replacement policy requires a separate versioned release approval."},
    ]
    # Two version identifiers refer to different scientific objects:
    # the version of the library that PRODUCED an archival baseline and the
    # version of the library that CONSUMES it. Their inequality is normal.
    # Neither a version string mismatch nor developer dependency mismatch
    # alone demonstrates a metric or GT incompatibility.
    compatibility_notes=[]
    if source != current:
        compatibility_notes.append({
            "code":"ARCHIVED_SOURCE_VERSION_DIFFERS_FROM_CONSUMER",
            "reason":"The frozen baseline records its producing library version, not a requirement on the current consumer. This is provenance, not a release blocker. Compatibility follows the scientific protocol, reference and observed numerical behavior.",
        })
    if version["status"]!="MATCH":
        compatibility_notes.append({
            "code":"CERTIFIED_DEPENDENCY_VERSIONS_DIFFER",
            "reason":"Dependency versions differ from the developer-certified environment; preserve and report this qualification. A passed numerical stability smoke check supports observed stability but does not prove complete equivalence.",
        })
    if stability["status"]!="PASSED":
        blockers.append({
            "code":"NUMERICAL_STABILITY_UNVERIFIED_OR_FAILED",
            "reason":"Independent PRNG/front stability tests did not all pass.",
        })
    return {
        "status":"BLOCKED_PENDING_SCIENTIFIC_RELEASE_REVIEW",
        "certified":False,"publishable":False,"overwrite_canonical_allowed":False,
        "evidence_status":"PAIRED_CANDIDATE_TESTED_NOT_HISTORICALLY_CERTIFIED",
        "calibration_provenance":{
            "original_baseline_version":source,
            "consumer_library_version":current,
            "original_baseline_sha256":manifest["source_baseline_sha256"],
            "candidate_baseline_sha256":manifest["candidate_baseline_sha256"],
            "candidate_protocol":manifest["candidate_method"],
            "proposed_measure":q_study["reference_measure"],
            "source_revision":manifest["source_revision"],
        },
        "environment":{
            "version_status":version["status"],
            "version_differences":version.get("differences",{}),
            "stability_status":stability["status"],
            "stability_checks":stability.get("checks",{}),
            "runtime":environment.get("runtime",{}),
        },
        "paired_q_impact":{
            "problem":q_study["problem"],
            "informative_fronts":q_study["informative_checkpoint_count"],
            "clipped_fronts":q_study["clipped_checkpoint_count"],
            "max_absolute_q_difference":q_study["max_informative_absolute_q_shift"],
            "allowable_drift_threshold":None,
            "why_no_threshold":"No retrospective scientific acceptance margin is justified by this exploratory pilot.",
        },
        "checks":checks,
        "release_blockers":blockers,
        "compatibility_notes":compatibility_notes,
        "version_difference_alone_requires_new_baseline":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--q-study",type=Path,required=True)
    p.add_argument("--historical-preflight",type=Path,required=True)
    p.add_argument("--environment",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() or out.is_relative_to(RESOURCES) or
        out.is_relative_to(REPORTS) or out==CANONICAL):
        p.error("Refusing overwrite or certified release destination")
    values=[json.loads(f.read_text()) for f in (
        a.manifest,a.q_study,a.historical_preflight,a.environment)]
    assessment=assess_release(*values)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(assessment,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":assessment["status"],
        "blockers":[x["code"] for x in assessment["release_blockers"]],
        "environment":assessment["environment"]["stability_status"],
        "source":assessment["calibration_provenance"]["original_baseline_version"],
        "consumer":assessment["calibration_provenance"]["consumer_library_version"],
    },indent=2))

if __name__=="__main__":
    main()
