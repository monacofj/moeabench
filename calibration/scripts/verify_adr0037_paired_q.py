#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired real-NSGA-II FAIR Q audit for ADR 0037's EXISTING PF sampling law.

Compare the frozen historical Closeness baseline and a *staged* correction
from the same underlying MOP analytical front sampler, without imposing
a uniform-area reference law. No installed baseline, report or GT is touched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import moeabench as mb
from moeabench.diagnostics import baselines

ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/"moeabench/diagnostics/resources/baselines_v0.14.0.json"
PROTECTED=(ROOT/"moeabench/diagnostics/resources").resolve()
REPORTS=(ROOT/"calibration/reports").resolve()
CANONICAL=(ROOT/"calibration/audit_report.html").resolve()


def _digest(values):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(values,dtype=np.float64)).tobytes()).hexdigest()


def paired_study(candidate, *, problem="DTLZ1", K=100, generations=500, seeds=(1,7)):
    if problem not in ("DTLZ1","DTLZ2"):
        raise ValueError("Require an independently analytically supported MOP pilot")
    if len(seeds)<2 or len(seeds)!=len(set(seeds)) or K<8 or generations<2:
        raise ValueError("Require two real optimizer seeds and a valid search budget")
    candidate=Path(candidate)
    source=json.loads(FROZEN.read_text())
    staged=json.loads(candidate.read_text())
    if staged.get("version")!=source.get("version"):
        raise ValueError("No archived baseline provenance retagging permitted")
    krec=staged["problems"][problem][str(K)]["closeness"]
    if (krec.get("ideal_res_method")!="independent_analytical_pf" or
        "reference_measure" in krec):
        raise ValueError("Expected existing MOP analytical PF law, not uniform area")
    eps=float(krec["ideal_res"])
    if not np.isfinite(eps) or eps<=0:
        raise ValueError("No positive finite correction available")
    results=[]
    for seed in seeds:
        exp=mb.experiment(
            mop=getattr(mb.mops,problem)(M=3),
            moea=mb.moeas.NSGA2(population=K,generations=generations),
        )
        exp.run(seed=int(seed),silent=True)
        archived=mb.clinic.audit(exp)
        with baselines.use_baselines(str(candidate)):
            revised=mb.clinic.audit(exp)
        if (archived.fr is None or revised.fr is None or
            archived.quality is None or revised.quality is None):
            raise AssertionError("Need real physical and clinical reports for both baselines")
        before=archived.diagnostic_context
        after=revised.diagnostic_context
        if (before.get("gt_source")!="registered" or
            after.get("gt_source")!="registered" or
            not np.array_equal(before["GT"],after["GT"]) or
            not np.array_equal(before["P_initial"],after["P_initial"]) or
            not np.array_equal(before["P_final"],after["P_final"]) or
            float(before["s_k"])!=float(after["s_k"])):
            raise AssertionError("Paired comparison altered real GT/P0/Pf/s_K")
        old_u=np.asarray(archived.fr.metrics["CLOSENESS"].raw_data,dtype=float)
        new_u=np.asarray(revised.fr.metrics["CLOSENESS"].raw_data,dtype=float)
        np.testing.assert_allclose(new_u,np.maximum(0.0,old_u-eps),
                                   atol=1e-12,rtol=0)
        same={}
        for metric in ("COVERAGE","GAP","REGULARITY","BALANCE","HEADWAY"):
            oq=float(archived.quality.scores["Q_"+metric].value)
            nq=float(revised.quality.scores["Q_"+metric].value)
            of=float(archived.fr.metrics[metric].value)
            nf=float(revised.fr.metrics[metric].value)
            np.testing.assert_allclose([oq,of],[nq,nf],atol=1e-12,rtol=0)
            same[metric]=True
        original_q=float(archived.quality.scores["Q_CLOSENESS"].value)
        candidate_q=float(revised.quality.scores["Q_CLOSENESS"].value)
        for v in (original_q,candidate_q):
            if not np.isfinite(v) or not 0<=v<=1:
                raise AssertionError("Invalid physical/clinical Q")
        results.append({
            "seed":int(seed),
            "P0_sha256":_digest(before["P_initial"]),
            "Pf_sha256":_digest(before["P_final"]),
            "GT_sha256":_digest(before["GT"]),
            "population_initial":len(before["P_initial"]),
            "population_final":len(before["P_final"]),
            "q_closeness_historical":original_q,
            "q_closeness_adr0037_candidate":candidate_q,
            "absolute_q_shift":abs(candidate_q-original_q),
            "both_endpoints_clipped":bool(
                (original_q==candidate_q==0) or
                (original_q==candidate_q==1)),
            "other_five_fair_and_q_unchanged":same,
        })
    return {
        "status":"NONCERTIFIED_REAL_PAIR_EXISTING_PF_LAW",
        "reference_law":"existing_mop_analytical_pf_independent_gt",
        "problem":problem,"K":K,"generations":generations,
        "frozen_sha256":hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "candidate_sha256":hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "ideal_res":eps,
        "runs":results,
        "max_absolute_q_closeness_shift":max(x["absolute_q_shift"] for x in results),
        "informative_final_runs":sum(not x["both_endpoints_clipped"] for x in results),
        "certified":False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--problem",choices=("DTLZ1","DTLZ2"),default="DTLZ1")
    parser.add_argument("--K",type=int,default=100)
    parser.add_argument("--generations",type=int,default=500)
    parser.add_argument("--seed",type=int,action="append",dest="seeds")
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    out=a.output.resolve()
    if (out.exists() or out.is_relative_to(PROTECTED) or
        out.is_relative_to(REPORTS) or out==CANONICAL):
        parser.error("Refusing existing or official output path")
    report=paired_study(a.candidate,problem=a.problem,K=a.K,
                        generations=a.generations,seeds=tuple(a.seeds or (1,7)))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],"problem":report["problem"],
        "ideal_res":report["ideal_res"],
        "max_absolute_q_closeness_shift":report["max_absolute_q_closeness_shift"],
        "informative_final_runs":report["informative_final_runs"],
    },indent=2))


if __name__=="__main__":
    main()
