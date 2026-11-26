# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only exact decision-image proof reach across actual DTLZ5 source pool.

One unchanged n=16 production PS call captures all 448 candidate decisions.
The classification operates on the exact binary rationals of each decision
and never infers Pareto membership from a finite Chebyshev gap. No post-hoc
snapping or production code/test/baseline modification is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from .dtlz5_final_selection_trace import trace_original_dtlz5_selection
from .dtlz56_paired_symbolic_audit import _row_symbolic_result
from .dtlz56_tight_sine_global_pf import (
    certify_tight_sine_global_pf, replay_tight_sine_global_pf,
)
from .dtlz5_row10_exact_dominance import (
    SOURCE_ROW_10, archived_row10_witness,
    certify_strict_dtlz5_m4_dominator,
)


def classify_original_decision(decision):
    """Three-valued exact statement about the *original decision image*."""
    v=tuple(float(x) for x in decision)
    if len(v)!=4 or any(not np.isfinite(x) or x<0 or x>1 for x in v):
        raise ValueError("Expected feasible original DTLZ5 M=4 K=1 decision")
    g=(Fraction.from_float(v[3])-Fraction(1,2))**2
    old=_row_symbolic_result(v,problem="DTLZ5",m=4,k=1)
    if old["status"]=="symbolic_decision_dominated":
        return "dominated", "exact_same_ray"
    if old["status"]=="symbolic_decision_global_pf":
        return "proved_pf", str(old["theorem"])
    if v == SOURCE_ROW_10:
        witness=certify_strict_dtlz5_m4_dominator(
            v,archived_row10_witness())
        if witness["conclusion"]=="rigorously_strictly_dominated":
            return "dominated","exact_off_ray_source_157"
    if g>0:
        cert=certify_tight_sine_global_pf(
            v[:3],g=g,problem="DTLZ5",k=1)
        if cert.conclusion=="rigorously_global_pf_member":
            if not replay_tight_sine_global_pf(
                v[:3],cert.to_dict(),g=g,problem="DTLZ5",k=1):
                raise AssertionError("Rational source certificate did not replay")
            return "proved_pf","tight_sine_global_support"
    return "unverified","insufficient_exact_theorem"


def _g_stratum(decision):
    g=(Fraction.from_float(float(decision[3]))-Fraction(1,2))**2
    if g==0:
        return "g_zero"
    if g<=Fraction(1,16):
        return "positive_g_up_to_1_16"
    if g<=Fraction(1,8):
        return "positive_g_1_16_to_1_8"
    return "positive_g_above_1_8"


def evaluate_proof_reach(source):
    """Classify exactly the captured pool; no GT generation in this function."""
    rows=source["candidate_pool_decision_hex"]
    families=source["candidate_pool_source_families"]
    if len(rows)!=448 or len(families)!=448 or len(source["selected_rows"])!=16:
        raise ValueError("Expected complete original n16 448-candidate trace")
    decisions=np.asarray([
        [float.fromhex(value) for value in row] for row in rows],dtype=float)
    results=[]
    for i,(decision,family) in enumerate(zip(decisions,families)):
        status,basis=classify_original_decision(decision)
        results.append({
            "index":i,"family":family,"g_stratum":_g_stratum(decision),
            "status":status,"basis":basis,
            "support_face_pattern":(
                "x2=x3=1" if decision[1]==1 and decision[2]==1 else
                "x2=1" if decision[1]==1 else
                "x3=1" if decision[2]==1 else
                "other"),
        })
    selected=[]
    for entry in source["selected_rows"]:
        decision=[float.fromhex(x) for x in entry["decision_hex"]]
        status,basis=classify_original_decision(decision)
        selected.append({
            "index":entry["selected_index"],"status":status,"basis":basis,
            "g_stratum":_g_stratum(decision),
            "candidate_index":(
                entry["trace"]["scalarizer_candidate_index"]
                if entry["trace"] is not None else None),
        })
    def counts(items):
        return {key:sum(x["status"]==key for x in items)
                for key in ("proved_pf","dominated","unverified")}
    by_family={
        family:counts([x for x in results if x["family"]==family])
        for family in ("sobol","gaussian","gamma")
    }
    by_stratum={
        key:counts([x for x in results if x["g_stratum"]==key])
        for key in ("g_zero","positive_g_up_to_1_16",
                    "positive_g_1_16_to_1_8","positive_g_above_1_8")
    }
    by_face={
        key:counts([x for x in results if x["support_face_pattern"]==key])
        for key in ("x2=x3=1","x2=1","x3=1","other")
    }
    return {
        "schema":"dtlz5_original_n16_pool_exact_proof_reach_v1",
        "problem":"DTLZ5","M":4,"K":1,
        "source_candidate_count":len(results),
        "source_selection_count":len(selected),
        "source_sha_from_trace":source.get("source_sha"),
        "pool_counts":counts(results),
        "by_source_family":by_family,
        "by_radial_stratum":by_stratum,
        "by_angular_support":by_face,
        "selected_counts":counts(selected),
        "selected_rows":selected,
        "candidate_rows":results,
        "current_proof_reach_not_complete_pf_oracle":True,
        "finite_gt_coverage":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Do not replace proof reach evidence")
    result=evaluate_proof_reach(
        trace_original_dtlz5_selection(include_candidate_pool=True))
    with args.report.open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({
        k:result[k] for k in (
            "pool_counts","by_source_family","by_radial_stratum",
            "by_angular_support","selected_counts","selected_rows",
            "scientific_certification",
        )},sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
