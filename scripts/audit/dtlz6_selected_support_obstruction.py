# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Failure-mode diagnosis of five original DTLZ6 selected unverified rows.

This is an interpretive research-only sidecar to the independently replayed
tight sine/root classifier, NOT an additional Pareto membership oracle.
It distinguishes a missing sufficient supporting angular pattern from a
supported pattern whose exact rational strict margin is nonpositive.
Identifies original exact binary decisions, angular boundary and radial
root enclosure. Never labels unverified rows dominated or proves the
completeness of any original positive-g Pareto component.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from fractions import Fraction
import numpy as np

from .dtlz6_tight_selected_source import (
    audit_dtlz6_tight_selected,replay_dtlz6_tight_selected,
)
from .dtlz56_exact_same_ray import _encode

SCHEMA="DTLZ6-M4K1-ORIGINAL-INCONCLUSIVE-SUPPORT-OBSTRUCTION-V1"


def explain_original_selected(decisions):
    base=audit_dtlz6_tight_selected(decisions)
    if not replay_dtlz6_tight_selected(decisions,base):
        raise ArithmeticError("Existing original source certificate did not replay")
    result=[]
    for r in base["rows"]:
        if r["historical_classifier_status"]!="unverified":
            continue
        proof=r["tight_sine_proof"]
        dec=tuple(Fraction.from_float(float.fromhex(v))
                  for v in r["decision_hex"])
        m=proof["strict_rational_margin"]
        if proof["supporting_objective"] is None:
            reason="no_available_sufficient_angular_support_pattern"
        elif m is None:
            reason="support_pattern_present_but_no_positive_radial_proof"
        else:
            margin=Fraction(int(m["numerator_hex"],16),
                            int(m["denominator_hex"],16))
            reason=("nonpositive_exact_support_margin" if margin<=0
                    else "positive_margin_requires_certification_reconciliation")
        result.append({
            "selected_index":r["index"],
            "decision_hex":r["decision_hex"],
            "x1_exact":_encode(dec[0]),
            "x2_boundary": "zero" if dec[1]==0 else "one" if dec[1]==1 else "open",
            "x3_boundary": "zero" if dec[2]==0 else "one" if dec[2]==1 else "open",
            "g_lower":proof["g_lower"],
            "g_upper":proof["g_upper"],
            "supporting_objective":proof["supporting_objective"],
            "sine_power":proof["sine_power"],
            "taylor_support_margin":m,
            "reason":reason,
            "new_membership":"rigorously_global_pf_member" if r[
                "new_taylor_support_proved"] else "unverified",
        })
    return {
        "schema":SCHEMA,"problem":"DTLZ6","M":4,"K":1,
        "selected_count":len(base["rows"]),
        "source_decision_sha256":base["source_decision_sha256"],
        "original_historical_inconclusive_indices":base[
            "historical_unverified_indices"],
        "newly_proved_historical_indices":base[
            "resolved_historical_unverified_indices"],
        "still_inconclusive_historical_indices":base[
            "still_unverified_historical_indices"],
        "row_diagnostics":result,
        "necessary_conditions_not_sufficient":True,
        "whole_pf_completeness":"unverified",
        "scientific_certification":"unverified",
    }


def replay_explanation(decisions,record):
    try:
        return record==explain_original_selected(decisions)
    except (TypeError,ValueError,KeyError,ArithmeticError,OverflowError):
        return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-bundle",required=True,type=Path)
    p.add_argument("--report",required=True,type=Path)
    args=p.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Refusing to overwrite research evidence")
    digest=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected original source bundle fields")
        if (str(data["problem"].item())!="DTLZ6" or
            int(data["M"].item())!=4 or int(data["K"].item())!=1):
            raise ValueError("Expected original DTLZ6 M4/K1")
        ps=data["decisions"].copy()
    report=explain_original_selected(ps)
    if not replay_explanation(ps,report):
        raise ArithmeticError("Exact original-source diagnosis failed replay")
    report["paired_artifact_sha256"]=digest
    with args.report.open("x",encoding="utf-8") as f:
        json.dump(report,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print("DTLZ6 historical inconclusive selected indices:",
          report["original_historical_inconclusive_indices"])
    for row in report["row_diagnostics"]:
        print("DTLZ6 support obstruction",row["selected_index"],
              "coordinates",row["decision_hex"],
              "boundary",row["x2_boundary"],row["x3_boundary"],
              "g in",row["g_lower"],row["g_upper"],
              "objective",row["supporting_objective"],
              "sine_power",row["sine_power"],
              "reason",row["reason"])
    print("DTLZ6 scientifically certified:",report["scientific_certification"])


if __name__=="__main__":
    main()
