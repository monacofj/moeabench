# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only provenance trace of actual original DTLZ6 n16 selected rows.

Observe the ORIGINAL scalarizer and decision reconstruction with temporary
Python wrappers (restored in finally). Check that each nonanchor selected
decision matches an actual reconstructed 448-candidate source row bitwise.
Record the exact angular active-set flags and bounded scalarization gap for
the newly proved strictly dominated original selected rows 7 and 11.
These are provenance diagnostics, NOT a proof of causal origin, a validated
production repair, or scientific acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from moeabench.mops import DTLZ6
from moeabench.mops import _dtlz56_front as front

SCHEMA="DTLZ6-ORIGINAL-N16-SELECTED-SCALARIZATION-TRACE-V1"


def trace_original_selected_dtlz6():
    mop=DTLZ6(M=4,K=1)
    original_solver=front.scalarized_candidate
    original_decision=front.decision_vector
    candidates=[]
    reconstructed=[]
    def capture_solver(*args,**kwargs):
        result=original_solver(*args,**kwargs)
        candidates.append(result)
        return result
    def capture_decision(candidate,**kwargs):
        result=original_decision(candidate,**kwargs)
        reconstructed.append(tuple(float(x) for x in result))
        return result
    try:
        front.scalarized_candidate=capture_solver
        front.decision_vector=capture_decision
        selected=np.asarray(mop.ps(n_points=16),dtype=float)
    finally:
        front.scalarized_candidate=original_solver
        front.decision_vector=original_decision
    if len(candidates)!=448 or len(reconstructed)!=448 or (
        selected.shape!=(16,4)
    ):
        raise AssertionError("Incomplete original 448-source selected trace")
    rows=[]
    for selected_index,decision in enumerate(selected):
        hits=[i for i,row in enumerate(reconstructed)
              if np.array_equal(np.asarray(row,dtype=float),decision)]
        if selected_index>=4 and not hits:
            raise AssertionError("Nonanchor selected decision absent from 448 pool")
        record={
            "selected_index":selected_index,
            "decision_hex":[float(x).hex() for x in decision],
            "matched_source_candidate_indices":hits,
            "gzero_anchor_prefix":selected_index<4,
        }
        if hits:
            index=hits[0]
            candidate=candidates[index]
            record["source_family"]=(
                "sobol" if index<256 else
                "gaussian" if index<384 else "gamma")
            record["scalarization_candidate_index"]=index
            record["scalarization_g"]=float(candidate.g).hex()
            record["scalarization_absolute_gap"]=float(
                candidate.abs_gap).hex()
            record["scalarization_lower_bound"]=float(
                candidate.lower_bound).hex()
            record["angular_active_faces"]=list(
                candidate.angular_active_faces
            ) if candidate.angular_active_faces is not None else None
        else:
            record["source_family"]="exact_gzero_anchor"
            record["scalarization_candidate_index"]=None
            record["scalarization_g"]=None
            record["scalarization_absolute_gap"]=None
            record["scalarization_lower_bound"]=None
            record["angular_active_faces"]=None
        rows.append(record)
    return {
        "schema":SCHEMA,"problem":"DTLZ6","M":4,"K":1,
        "unchanged_production_original_ps":True,
        "source_candidate_count":len(candidates),
        "selected_count":len(rows),
        "source_selected_decisions_sha256":hashlib.sha256(
            b"".join(float(v).hex().encode("ascii")+b";"
                     for row in selected for v in row)
        ).hexdigest(),
        "known_independently_dominated_selected_indices":[7,11],
        "specific_provenance_rows":[rows[7],rows[11]],
        "all_selected_provenance_rows":rows,
        "causal_repair_of_sampler":"unverified",
        "whole_gt_membership":"rejected_by_independent_exact_proof",
        "finite_indicator_suitability":"unverified",
        "scientific_certification":"unverified",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args(argv)
    if a.report.exists():
        raise FileExistsError("Do not overwrite research evidence")
    r=trace_original_selected_dtlz6()
    with a.report.open("x",encoding="utf-8") as f:
        json.dump(r,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    for record in r["specific_provenance_rows"]:
        print("Confirmed exact-dominated selected index",record["selected_index"],
              "source candidate index",record["scalarization_candidate_index"],
              "source family",record["source_family"],
              "original angular active faces",record["angular_active_faces"],
              "original scalarization absolute gap",
              record["scalarization_absolute_gap"],
              "exact selected decision",record["decision_hex"])
    print("Source selected decision SHA256",
          r["source_selected_decisions_sha256"])
    print("Causality",r["causal_repair_of_sampler"])


if __name__=="__main__":
    main()
