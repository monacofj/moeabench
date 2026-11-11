#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #75: auditable 14-MOP PF-source validity screen, NOT certification.

The previous 528-record study checked *geometrical overlap* with GT and
independence of analytical points. Those properties do not establish that
the analytical sampler produces genuine globally Pareto-optimal points or
a justified probability law. This script adds sampled nondominance,
feasibility, branch/degeneracy checks and explicit *human-review statuses*.

Never modify a frozen baseline, GT package, or audit_report.
"""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from moeabench import mops
from calibration.scripts.generate_baselines import _analytical_ideal_from_package

BASELINE=ROOT/"moeabench/diagnostics/resources/baselines_v0.14.0.json"
REFERENCES=ROOT/"moeabench/diagnostics/resources/references"
PROTECTED=(ROOT/"moeabench/diagnostics/resources").resolve()
REPORTS=(ROOT/"calibration/reports").resolve()
OFFICIAL_REPORT=(ROOT/"calibration/audit_report.html").resolve()

# Sampling laws are sourced from the reviewed MOP.ps/pf implementations,
# not reverse-engineered from near-neighbor distances.
LAWS={
    "DPF1":("linspace_base_decision_parameter","degenerate_1d_curve"),
    "DPF2":("linspace_base_decision_parameter","potentially_disconnected_dtlz7_like"),
    "DPF3":("linspace_base_decision_parameter_then_NDS_filter","variable_cardinality_chaotic_projection"),
    "DPF4":("linspace_base_decision_parameter","piecewise_chaotic_projection"),
    "DPF5":("meshgrid_x1_x2_with_x1_gt_one_third_piecewise_branch","2d_decision_grid_and_piecewise_objective_map"),
    "DTLZ1":("fixed_rng_uniform_decision_parameters","nonlinear_objective_simplex_pushforward"),
    "DTLZ2":("positive_gaussian_sphere_directions_plus_axis_extrema","positive_hypersphere"),
    "DTLZ3":("inherited_DTLZ2_positive_sphere_directions","positive_hypersphere"),
    "DTLZ4":("inverse_power_100_positive_sphere_directions","positive_hypersphere"),
    "DTLZ5":("fixed_rng_uniform_decision_parameters","degenerate_1d_PF"),
    "DTLZ6":("fixed_rng_uniform_decision_parameters","degenerate_1d_PF"),
    "DTLZ7":("fixed_rng_uniform_decision_parameters_no_PF_filter","disconnected_PF"),
    "DTLZ8":("static_optimal_x_csv_or_guided_analytical_solver","constrained_heterogeneous_sampling"),
    "DTLZ9":("linspace_last_objective_then_inverse_to_decisions","constrained_1d_curve"),
}

def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(array,dtype=np.float64)).tobytes()).hexdigest()


def _nd_evidence(objectives):
    """Within-sample ND check: necessary evidence, NOT a proof of global Pareto."""
    objectives=np.asarray(objectives,dtype=np.float64)
    if objectives.ndim!=2 or len(objectives)<2 or not np.all(np.isfinite(objectives)):
        raise ValueError("Invalid finite PF sample")
    unique=np.unique(objectives,axis=0)
    indices=NonDominatedSorting().do(unique,only_non_dominated_front=True)
    return {
        "points":int(len(objectives)),
        "unique_points":int(len(unique)),
        "duplicate_points":int(len(objectives)-len(unique)),
        "within_sample_dominated":int(len(unique)-len(indices)),
        "within_sample_nondominated_fraction":float(len(indices)/len(unique)),
    }


def classify(name, *, analytical_supported, candidates, feasibility, branch):
    """Conservative status. A screen never auto-certifies a probability law."""
    if name not in LAWS:
        raise ValueError("Unknown MOP")
    if not analytical_supported:
        return "NO_COMPATIBLE_INDEPENDENT_ANALYTICAL_PF"
    if feasibility is not None and feasibility["infeasible"]>0:
        return "HOLD_INFEASIBLE_ANALYTICAL_POINTS"
    if candidates["within_sample_dominated"]>0:
        return "HOLD_DOMINATED_ANALYTICAL_POINTS"
    if name=="DPF5":
        return "HOLD_DPF5_PIECEWISE_BRANCH_AND_OBJECTIVE_DENSITY_UNPROVEN"
    if name in ("DTLZ7","DPF2"):
        return "HOLD_DISCONNECTED_FRONT_SUPPORT_UNPROVEN"
    if name in ("DTLZ5","DTLZ6","DTLZ9"):
        return "CONDITIONAL_DEGENERATE_CURVE_LAW_NEEDS_REVIEW"
    if name=="DTLZ8":
        return "HOLD_CONSTRAINED_STATIC_REFERENCE_PROVENANCE_UNPROVEN"
    return "CONDITIONAL_GEOMETRY_ONLY_NOT_PROBABILITY_CERTIFIED"


def _physical_sample(mop,n_points):
    """Use actual MOP.ps() and evaluation; don't infer objective-law from GT."""
    decision=np.asarray(mop.ps(n_points=n_points),dtype=np.float64)
    if decision.ndim!=2 or not len(decision) or not np.all(np.isfinite(decision)):
        raise ValueError("Invalid analytical Pareto-set sampler")
    inequality=mop.get_n_ieq_constr()
    evaluated=mop.evaluation(decision,n_ieq_constr=inequality)
    physical=np.asarray(evaluated["F"],dtype=np.float64)
    if physical.shape!=(len(decision),mop.M):
        raise ValueError("Incorrect objective shape")
    feasible=None
    if inequality:
        constraints=np.asarray(evaluated["G"],dtype=np.float64)
        if constraints.shape[0]!=len(decision):
            raise ValueError("Incorrect constraint size")
        feasible={
            "constraint_count":int(inequality),
            "infeasible":int(np.count_nonzero(np.any(constraints>1e-10,axis=1))),
            "total":int(len(decision)),
            "maximum_violation":float(max(0.0,np.max(constraints))),
        }
    branch=None
    if mop.__class__.__name__=="DPF5":
        mask=decision[:,0]>1.0/3.0
        unique=np.unique(physical,axis=0)
        unique_P=np.unique(physical[~mask],axis=0)
        unique_D=np.unique(physical[mask],axis=0)
        nd=NonDominatedSorting().do(unique,only_non_dominated_front=True)
        # Set D depends only on x1; the M=3 decision-grid's x2 values
        # collapse to repeated objective vectors. The MOP-ps pushforward
        # weights set D VERY differently from the unique-objective pool
        # actually used by independent_analytical_reference().
        n_P,n_D=int(np.count_nonzero(~mask)),int(np.count_nonzero(mask))
        u_P,u_D=len(unique_P),len(unique_D)
        p_raw=float(n_D/(n_P+n_D))
        p_unique=float(u_D/(u_P+u_D))
        branch={
            "threshold":"x1 > 1/3",
            "decision_grid_rows_set_P":n_P,
            "decision_grid_rows_set_D":n_D,
            "unique_objective_rows_set_P":int(u_P),
            "unique_objective_rows_set_D":int(u_D),
            "set_P_fraction":float(np.mean(~mask)),
            "set_D_fraction":p_raw,
            "set_D_fraction_after_unique_objective_rows":p_unique,
            "set_D_raw_to_unique_probability_ratio":float(p_raw/p_unique),
            "whole_sample_nd_unique_points":int(len(nd)),
            "objective_regions_have_distinct_conditional_formulas":True,
            "warning":"Removing objective duplicates changes the implied sampling probability measure",
        }
    return physical,feasible,branch


def audit_mop(name, baseline, *, n_points=1024, package_root=REFERENCES):
    if name not in LAWS:
        raise ValueError("Unknown MOP "+str(name))
    model=getattr(mops,name)()
    pkgpath=Path(package_root)/name/"calibration_package.npz"
    if not pkgpath.is_file():
        raise ValueError("Frozen GT package absent for "+name)
    with np.load(pkgpath,allow_pickle=False) as pkg:
        gt=np.asarray(baseline["problems"][name]["gt_reference"],dtype=np.float64)
        pkg_gt=np.asarray(pkg["gt_norm"],dtype=np.float64)
        if not np.array_equal(gt,pkg_gt):
            raise ValueError("Frozen baseline and package GT mismatch: "+name)
        ideal=np.asarray(pkg["ideal"],dtype=np.float64)
        nadir=np.asarray(pkg["nadir"],dtype=np.float64)
        fresh=_analytical_ideal_from_package(name,gt,pkg)

    raw,feasibility,branch=_physical_sample(model,n_points)
    physical_mop_pf=np.asarray(model.pf(n_points=n_points),dtype=np.float64)
    if physical_mop_pf.ndim!=2 or physical_mop_pf.shape[1]!=gt.shape[1]:
        raise ValueError("MOP.pf returned invalid objective shape: "+name)
    span=nadir-ideal
    if np.any(span<=0):
        raise ValueError("Invalid frozen normalization span")
    normalized=(physical_mop_pf-ideal)/span
    tree=cKDTree(gt)
    local_spacing=float(np.median(tree.query(gt,k=2)[0][:,1]))
    if not np.isfinite(local_spacing) or local_spacing<=0:
        raise ValueError("Invalid GT sampling resolution")
    offsets=tree.query(normalized,k=1)[0]
    candidates=_nd_evidence(physical_mop_pf)
    raw_evidence=_nd_evidence(raw)
    name_law,geometry=LAWS[name]
    code_path=Path(inspect.getfile(model.__class__))
    report={
        "problem":name,
        "objective_count":int(gt.shape[1]),
        "frozen_gt_points":int(len(gt)),
        "frozen_gt_sha256":_sha(gt),
        "mop_source_sha256":hashlib.sha256(code_path.read_bytes()).hexdigest(),
        "PF_sampling_law":name_law,
        "expected_structure":geometry,
        "requested_sample_size":int(n_points),
        "mop_pf_cardinality":int(len(physical_mop_pf)),
        "sampled_pf":candidates,
        "raw_ps_evaluation":raw_evidence,
        "feasibility":feasibility,
        "DPF5_branch":branch,
        "distance_to_frozen_GT_in_local_spacing_units":{
            "median":float(np.median(offsets)/local_spacing),
            "p95":float(np.percentile(offsets,95)/local_spacing),
        },
        "independent_analytical_sample_available":bool(fresh is not None),
        "independent_points":int(len(fresh)) if fresh is not None else 0,
        "independence_verified_to_frozen_GT": bool(
            fresh is not None and np.all(tree.query(fresh,k=1)[0]>1e-10)),
        "independent_PF_nd_check":_nd_evidence(fresh) if fresh is not None else None,
        "normative_probability_law_approved":False,
        "global_Pareto_optimality_proven":False,
    }
    if fresh is not None:
        report["independent_PF_sha256"]=_sha(fresh)
    report["review_status"]=classify(
        name,analytical_supported=bool(fresh is not None),
        candidates=(report["independent_PF_nd_check"] or candidates),
        feasibility=feasibility,branch=branch,
    )
    return report


def run_audit(original, *, n_points=1024, package_root=REFERENCES):
    if set(original["problems"])!=set(LAWS):
        raise ValueError("Expected exactly the fourteen archived MOPs")
    if not isinstance(n_points,int) or n_points<64 or n_points>2048:
        raise ValueError("Need bounded analytical sample [64,2048]")
    out={}
    for name in sorted(LAWS):
        out[name]=audit_mop(name,original,n_points=n_points,
                            package_root=package_root)
        print(name,out[name]["review_status"],
              "ND",out[name]["sampled_pf"]["within_sample_nondominated_fraction"],
              "rawND",out[name]["raw_ps_evaluation"]["within_sample_nondominated_fraction"],
              "independent",out[name]["independent_analytical_sample_available"],
              flush=True)
    return {
        "status":"NONCERTIFIED_PF_SOURCE_AUDIT_NEVER_BASELINE_CERTIFICATION",
        "MOP_count":len(out),
        "reviewed_source":"actual_MOP_ps_pf_and_frozen_GT",
        "nondominance_test":"within_sample_only_not_global_Pareto_proof",
        "geometry_test":"near_frozen_GT_not_probability_equivalence",
        "no_baseline_modified":True,
        "summary":{
            "independent_analytical_sources":sum(x["independent_analytical_sample_available"] for x in out.values()),
            "unsupported":sorted(k for k,v in out.items() if not v["independent_analytical_sample_available"]),
            "held_for_review":sorted(k for k,v in out.items() if v["review_status"].startswith("HOLD")),
            "conditionally_screened":sorted(k for k,v in out.items() if v["review_status"].startswith("CONDITIONAL")),
        },
        "problems":out,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=BASELINE)
    parser.add_argument("--points",type=int,default=1024)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    out=a.output.resolve()
    if (out.exists() or out==a.baseline.resolve() or out.is_relative_to(PROTECTED)
            or out.is_relative_to(REPORTS) or out==OFFICIAL_REPORT):
        parser.error("Refusing overwrite or official baseline/report directory")
    original=json.loads(a.baseline.read_text())
    audit=run_audit(original,n_points=a.points)
    audit["baseline_sha256"]=hashlib.sha256(a.baseline.read_bytes()).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY",json.dumps(audit["summary"],sort_keys=True))


if __name__=="__main__":
    main()
