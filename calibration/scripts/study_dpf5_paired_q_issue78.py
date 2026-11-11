#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare three UNADOPTED DPF5 reference laws on identical real NSGA-II runs.

Only K=100's Closeness ECDF/residue are varied in memory. Historical GT, P0,
Pf, all other FAIR metrics and their calibration records are fixed. No baseline
is installed, and none of the three alternative probability laws is endorsed.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import moeabench as mb

from moeabench.diagnostics import baselines
from calibration.scripts.stage_closeness_adr0037 import revise_closeness_record
from calibration.scripts.study_dpf5_branch_measure_issue78 import (
    ROOT, FROZEN, RESOURCES, REPORTS, OFFICIAL, run,
)


def _hash(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(
        a,dtype=np.float64)).tobytes()).hexdigest()


def candidate_for_law(source, k, epsilon, measure_name):
    """Build a counterfactual *in-memory* Closeness-only reference."""
    if not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError("Counterfactual correction must be positive and finite")
    if measure_name not in (
        "uniform_decision_grid_pushforward",
        "uniform_distinct_objective_vectors",
        "illustrative_equal_branch_mixture",
    ):
        raise ValueError("Unsupported DPF5 probability law")
    candidate=copy.deepcopy(source)
    old=source["problems"]["DPF5"][str(k)]["closeness"]
    updated=revise_closeness_record(old,float(epsilon))
    updated["ideal_res_method"]="noncertified_DPF5_"+measure_name
    candidate["problems"]["DPF5"][str(k)]["closeness"]=updated
    return candidate


def study(*,side=128,k=100,generations=300,seeds=(1,7)):
    if (len(seeds)<2 or len(seeds)!=len(set(seeds)) or
            k<8 or generations<2):
        raise ValueError("Require two distinct real-MOEA seeds and valid budget")
    sensitivity=run((side,),(k,))
    row=sensitivity["grids"][str(side)]["K"][str(k)]
    eps=row["ideal_res_by_measure"]
    source=json.loads(FROZEN.read_text())
    variants={name:candidate_for_law(source,k,epsilon,name)
              for name,epsilon in eps.items()}
    results=[]
    for seed in seeds:
        exp=mb.experiment(mop=mb.mops.DPF5(D=2,M=3),
                          moea=mb.moeas.NSGA2(population=k,generations=generations))
        exp.run(seed=int(seed),silent=True)
        old=mb.clinic.audit(exp)
        if old.fr is None or old.quality is None:
            raise AssertionError("Historical DPF5 clinical audit unavailable")
        ctx=old.diagnostic_context
        if ctx["gt_source"]!="registered":
            raise AssertionError("Paired experiment must use original frozen GT")
        old_cl=float(old.quality.scores["Q_CLOSENESS"].value)
        old_dist=np.asarray(old.fr.metrics["CLOSENESS"].raw_data,dtype=float)
        trials={}
        for name,candidate in variants.items():
            with baselines.use_baselines(candidate):
                alt=mb.clinic.audit(exp)
            if alt.fr is None or alt.quality is None:
                raise AssertionError("Counterfactual paired audit unavailable")
            newctx=alt.diagnostic_context
            for array in ("GT","P_initial","P_final"):
                if not np.array_equal(ctx[array],newctx[array]):
                    raise AssertionError("Changed paired "+array)
            if ctx["s_k"]!=newctx["s_k"]:
                raise AssertionError("Changed paired frozen resolution")
            eps_val=eps[name]
            np.testing.assert_allclose(
                alt.fr.metrics["CLOSENESS"].raw_data,
                np.maximum(0.,old_dist-eps_val),
                atol=1e-12,rtol=0)
            for metric in ("COVERAGE","GAP","REGULARITY","BALANCE","HEADWAY"):
                np.testing.assert_allclose(
                    [alt.fr.metrics[metric].value,
                     alt.quality.scores["Q_"+metric].value],
                    [old.fr.metrics[metric].value,
                     old.quality.scores["Q_"+metric].value],
                    atol=1e-12,rtol=0)
            q=float(alt.quality.scores["Q_CLOSENESS"].value)
            if not 0<=q<=1 or not np.isfinite(q):
                raise AssertionError("Invalid clinical Q under alternate law")
            trials[name]={"epsilon":float(eps_val),
                          "q_closeness":q,
                          "abs_shift_from_historical":abs(q-old_cl),
                          "physical_closeness":float(alt.fr.metrics["CLOSENESS"].value),
                          "other_five_fair_and_q_unchanged":True}
        # Confirm use_baselines has restored the original cached source.
        again=mb.clinic.audit(exp)
        np.testing.assert_allclose(
            again.quality.scores["Q_CLOSENESS"].value,old_cl,
            atol=1e-12,rtol=0)
        all_q=[v["q_closeness"] for v in trials.values()]
        results.append({
            "seed":int(seed),
            "P0_sha256":_hash(ctx["P_initial"]),
            "Pf_sha256":_hash(ctx["P_final"]),
            "GT_sha256":_hash(ctx["GT"]),
            "original_q_closeness":old_cl,
            "counterfactuals":trials,
            "max_between_laws_q_closeness_difference":float(max(all_q)-min(all_q)),
            "informative_between_laws":len(set(round(x,12) for x in all_q))>1,
        })
    return {
        "status":"NONCERTIFIED_DPF5_REAL_MOEA_PAIRED_Q_SENSITIVITY",
        "problem":"DPF5","K":int(k),"generations":int(generations),
        "grid_side":int(side),
        "frozen_baseline_sha256":hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "reference_laws_are_counterfactual":True,
        "changed_only_closeness_in_memory":True,
        "all_other_metrics_unchanged":True,
        "baseline_files_modified":False,
        "law_epsilons":eps,
        "seeds":list(map(int,seeds)),
        "runs":results,
        "maximum_between_laws_q_shift":max(
            r["max_between_laws_q_closeness_difference"] for r in results),
        "certified":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side",type=int,default=128)
    p.add_argument("--K",type=int,default=100)
    p.add_argument("--generations",type=int,default=300)
    p.add_argument("--seed",type=int,action="append",dest="seeds")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() or out.is_relative_to(RESOURCES) or
            out.is_relative_to(REPORTS) or out==OFFICIAL):
        p.error("Refusing existing/official scientific output")
    report=study(side=a.side,k=a.K,generations=a.generations,
                 seeds=tuple(a.seeds or (1,7)))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "generations":report["generations"],
        "max_between_laws_q_shift":report["maximum_between_laws_q_shift"],
        "runs":[{
            "seed":x["seed"],
            "old_q":x["original_q_closeness"],
            "q_by_law":{k:v["q_closeness"] for k,v in x["counterfactuals"].items()},
            "q_spread":x["max_between_laws_q_closeness_difference"],
        } for x in report["runs"]]
    },indent=2))


if __name__=="__main__":
    main()
