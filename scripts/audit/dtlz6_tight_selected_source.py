# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-bound decision-image ledger for original selected DTLZ6 M4/K1 n16.

Opt-in independent comparison of historical coarse sine<=theta irrational
root support certificate and new sharper exact Taylor-sine proof.
A proof replay can turn 'unverified' into mathematically proved PF
membership; an inconclusive result is NOT domination, and neither
proves finite stored F or full continuous GT indicator fitness.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
from .dtlz56_dtlz6_radial_enclosure import (
    certify_dtlz6_radial_interval_pf,replay_dtlz6_radial_interval_pf,
)
from .dtlz6_pool_proof_reach import classify_dtlz6_decision
from .dtlz6_tight_irrational_pf import (
    certify_dtlz6_tight_radial_pf,replay_dtlz6_tight_radial_pf,
)

SCHEMA="DTLZ6-M4K1-EXACT-TIGHT-SINE-SELECTED-SOURCE-LEDGER-V1"


def audit_dtlz6_tight_selected(decisions,*,root_bits=64,pi_terms=16):
    try:
        rows=tuple(tuple(Fraction.from_float(float(v)) for v in row)
                   for row in decisions)
    except (TypeError,ValueError,OverflowError) as ex:
        raise ValueError("Expected feasible finite selected PS") from ex
    if not rows or any(len(row)!=4 or any(not 0<=v<=1 for v in row)
                       for row in rows):
        raise ValueError("Expected original selected M4 K1 PS")
    result=[]
    for i,row in enumerate(rows):
        old=certify_dtlz6_radial_interval_pf(
            row[:3],row[3:],k=1,root_bits=root_bits,pi_terms=pi_terms)
        tight=certify_dtlz6_tight_radial_pf(
            row[:3],row[3:],k=1,root_bits=root_bits,pi_terms=pi_terms)
        op=old.conclusion=="rigorously_global_pf_member"
        tp=tight["conclusion"]=="rigorously_global_pf_member"
        if op and not replay_dtlz6_radial_interval_pf(
            row[:3],row[3:],old.to_dict(),k=1,
            root_bits=root_bits,pi_terms=pi_terms):
            raise ArithmeticError("Existing exact radial proof failed replay")
        if tp and not replay_dtlz6_tight_radial_pf(
            row[:3],row[3:],tight,k=1,
            root_bits=root_bits,pi_terms=pi_terms):
            raise ArithmeticError("New exact Taylor proof failed replay")
        if op and not tp:
            raise ArithmeticError("Improved bound cannot lose original proof")
        # Existing g=0 unit-radius theorem is independent of positive g.
        is_gzero=row[-1]==0
        historical_status,historical_basis=classify_dtlz6_decision(row)
        if historical_status=="proved_pf" and not (tp or is_gzero):
            raise ArithmeticError("New exact support lost historical PF proof")
        result.append({
            "index":i,
            "decision_hex":[float(v).hex() for v in row],
            "old_radial_support_proved":op,
            "new_taylor_support_proved":tp,
            "gzero_global_norm_proved":is_gzero,
            "historical_classifier_status":historical_status,
            "historical_classifier_basis":historical_basis,
            "classification":(
                "rigorously_global_pf_member" if (tp or is_gzero)
                else "unverified"
            ),
            "tight_sine_proof":tight,
        })
    count=sum(x["classification"]=="rigorously_global_pf_member"
              for x in result)
    historical_unverified=[r["index"] for r in result
        if r["historical_classifier_status"]=="unverified"]
    resolved_historical=[r["index"] for r in result
        if r["historical_classifier_status"]=="unverified"
        and r["classification"]=="rigorously_global_pf_member"]
    return {
        "schema":SCHEMA,"problem":"DTLZ6","M":4,"K":1,
        "source_selected_n":len(rows),
        "root_bits":root_bits,"pi_terms":pi_terms,
        "source_decision_sha256":hashlib.sha256(
            b"".join(float(v).hex().encode("ascii")+b";"
                     for row in rows for v in row)).hexdigest(),
        "new_taylor_proved_indices":[r["index"] for r in result
                                     if r["new_taylor_support_proved"]],
        "newly_proved_over_coarse_indices":[r["index"] for r in result
            if r["new_taylor_support_proved"] and
               not r["old_radial_support_proved"]],
        "still_unverified_indices":[r["index"] for r in result
                                     if r["classification"]=="unverified"],
        "proved_selected_count":count,
        "historical_unverified_indices":historical_unverified,
        "resolved_historical_unverified_indices":resolved_historical,
        "still_unverified_historical_indices":[i for i in historical_unverified
            if i not in resolved_historical],
        "rows":result,
        "full_original_pf_completeness":"unverified",
        "finite_objective_F_alignment":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def replay_dtlz6_tight_selected(decisions,report,*,root_bits=64,pi_terms=16):
    try:
        return report==audit_dtlz6_tight_selected(
            decisions,root_bits=root_bits,pi_terms=pi_terms)
    except (ValueError,TypeError,KeyError,ArithmeticError,OverflowError):
        return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired-bundle",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args(argv)
    if a.report.exists():
        raise FileExistsError("Write-once research evidence required")
    digest=hashlib.sha256(a.paired_bundle.read_bytes()).hexdigest()
    with np.load(a.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Wrong original paired source archive schema")
        if str(data["problem"].item())!="DTLZ6" or (
            int(data["M"].item())!=4 or int(data["K"].item())!=1
        ):
            raise ValueError("Expected original DTLZ6 M4/K1")
        ps=data["decisions"].copy()
    out=audit_dtlz6_tight_selected(ps)
    if not replay_dtlz6_tight_selected(ps,out):
        raise AssertionError("Independent Taylor source ledger replay failed")
    out["paired_artifact_sha256"]=digest
    with a.report.open("x",encoding="utf-8") as f:
        json.dump(out,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("DTLZ6 improved exact irrational-root Taylor proved selected indices",
          out["newly_proved_over_coarse_indices"])
    print("DTLZ6 resolved historical unverified selected indices",
          out["resolved_historical_unverified_indices"])
    print("DTLZ6 remaining historical unverified selected indices",
          out["still_unverified_historical_indices"])
    print("DTLZ6 remaining unverified selected indices",
          out["still_unverified_indices"])
    print("DTLZ6 whole scientific certification",out["scientific_certification"])


if __name__=="__main__":
    main()
