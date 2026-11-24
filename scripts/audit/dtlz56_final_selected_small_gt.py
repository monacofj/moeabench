# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-bound analysis of the actual *final selected* DTLZ5/6 M=4 PS.

Unlike the fixed-ray studies, this invokes the unchanged production
mop.ps(n_points=16) ONCE per problem: all candidate families, the full
bounded Chebyshev solver, g=0 anchors, and final greedy maximin objective
selection. Consequently this diagnostic is explicitly opt-in and never
runs during ordinary unit tests.

Writes a separate, immutable paired PS/F research artifact and a JSON
geometry/conditional-proof report, both in exclusive-create mode. The
16 selected rows are exactly the output of the existing public sampler;
none are silently snapped or replaced. The report is observational,
NOT a proof of total PF coverage, finite IGD/HV adequacy or ADR0059 GT
certification. A deterministic full pool costs at least 448 scalarized
solves per original problem, independent of the requested 16 output rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np

from .dtlz56_paired_gt_evidence import (
    _original_objectives, save_paired_candidate, verify_paired_candidate,
)
from .dtlz56_paired_symbolic_audit import audit_symbolic_candidate
from .dtlz56_dtlz6_radial_enclosure import (
    audit_dtlz6_irrational_radial_rows,
)
from .dtlz56_dtlz6_support_proximity import (
    certify_dtlz6_support_proximity,
)


SCHEMA = "moeabench_dtlz56_final_selected_small_gt_audit_v1"
EXPECTED_N = 16


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except (OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return "unavailable"


def analyze_final_selected(decisions, objectives, *, problem, m, k,
                           tolerance=1e-9):
    """Inspect an already generated PS/F pair: no GT creation or writes.

    Original published equations are independently evaluated, and the
    existing additive symbolic decision-image predicates are applied.
    All distance/diversity statistics are observational and cannot
    establish global Pareto membership or whole-reference adequacy.
    """
    from .dtlz56_paired_gt_evidence import audit_paired_candidate
    aligned = audit_paired_candidate(
        decisions, objectives, problem=problem, m=m, k=k,
        tolerance=tolerance,
    )
    if aligned["independent_objective_alignment"] != "pass":
        raise ValueError("Independent original PS/F alignment failed")

    x = np.asarray(decisions, dtype=float)
    f = np.asarray(objectives, dtype=float)
    _, g = _original_objectives(x, problem=problem, m=m, k=k)
    n = len(x)
    if n < 2:
        raise ValueError("Geometry diagnostics require at least two rows")
    norms = np.linalg.norm(f,axis=1)
    # Nonnegative original spherical model:  ||F||_2 = 1+g.
    norm_residual = float(np.max(np.abs(norms-(1+g))))
    spans = np.ptp(f,axis=0)
    scales = np.maximum(spans,1e-12)
    geometry = (f[:,None,:]-f[None,:,:])/scales
    pairwise = np.linalg.norm(geometry,axis=2)
    np.fill_diagonal(pairwise,np.inf)
    nearest=np.min(pairwise,axis=1)
    q=np.quantile(g,[0,.25,.5,.75,1])
    exact_zero = g == 0.0
    positive = ~exact_zero
    positive_values=g[positive]

    theoretical = audit_symbolic_candidate(
        x, f, problem=problem, m=m, k=k,
        tolerance=tolerance,
    )
    irrational=None
    proximity=None
    if problem=="DTLZ6":
        irrational=audit_dtlz6_irrational_radial_rows(
            x,f,m=m,k=k,tolerance=tolerance,
        )
        proximity=[]
        for row_index,row in enumerate(x):
            cert=certify_dtlz6_support_proximity(
                row[:m-1],row[m-1:],k=k,
            )
            proximity.append({
                "row":row_index,
                "status":cert.conclusion,
                "distance_upper":cert.certified_l2_distance_upper,
                "supporting_objective":cert.supporting_objective,
            })

    # The results are measurements of the selected points, not a
    # measure of unknown continuous-front coverage.
    return {
        "schema":SCHEMA,
        "problem":problem,"M":m,"K":k,"n_points":n,
        "source": "unchanged_production_mop.ps_final_maximin_selection",
        "radial": {
            "exact_numeric_zero_count":int(np.count_nonzero(exact_zero)),
            "positive_count":int(np.count_nonzero(positive)),
            "quantiles":[float(v) for v in q],
            "positive_g_values":[float(v) for v in positive_values],
            "positive_row_indices":np.flatnonzero(positive).tolist(),
            "minimum_positive_g":(float(np.min(positive_values))
                                  if len(positive_values) else None),
            "maximum_positive_g":(float(np.max(positive_values))
                                  if len(positive_values) else None),
        },
        "geometry_observational":{
            "objective_component_min":np.min(f,axis=0).tolist(),
            "objective_component_max":np.max(f,axis=0).tolist(),
            "objective_component_span":spans.tolist(),
            "objective_range_normalized_nearest_neighbor_min":
                float(np.min(nearest)),
            "objective_range_normalized_nearest_neighbor_median":
                float(np.median(nearest)),
            "objective_range_normalized_nearest_neighbor_max":
                float(np.max(nearest)),
            "max_original_spherical_norm_residual":norm_residual,
            "max_original_equation_residual":
                aligned["max_abs_objective_error"],
        },
        "exact_rational_g_decision_image_ledger":theoretical,
        "dtlz6_irrational_g_support_ledger":irrational,
        "dtlz6_certified_pf_proximity":proximity,
        "whole_gt_global_membership":"unverified",
        "continuous_front_coverage":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def capture_and_audit(*, problem, m, k, n_points, bundle, report,
                       tolerance=1e-9):
    """One explicit original PS call, one evaluation, immutable outputs."""
    if problem not in ("DTLZ5","DTLZ6") or type(m) is not int or m!=4:
        raise ValueError("This bounded experiment requires original M=4")
    if type(k) is not int or k!=1:
        raise ValueError("This bounded experiment requires K=1")
    if type(n_points) is not int or n_points!=EXPECTED_N:
        raise ValueError("This bounded experiment requires exactly 16 points")
    if not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError("Tolerance must be finite and positive")
    if Path(bundle).exists() or Path(report).exists():
        raise FileExistsError("Refusing to overwrite paired evidence or report")
    from moeabench.mops import DTLZ5,DTLZ6
    mop=(DTLZ5 if problem=="DTLZ5" else DTLZ6)(M=m,K=k)
    # The production method includes its full source ray pool,
    # bounded scalarization and final greedy maximin selection.
    ps=np.asarray(mop.ps(n_points=n_points),dtype=float)
    if ps.shape!=(n_points,m+k-1):
        raise AssertionError("Production PS returned unexpected shape")
    f=np.asarray(mop.evaluation(ps)["F"],dtype=float)
    analyzed=analyze_final_selected(
        ps,f,problem=problem,m=m,k=k,tolerance=tolerance,
    )
    saved=save_paired_candidate(
        bundle,ps,f,problem=problem,m=m,k=k,tolerance=tolerance,
    )
    # Verify the actual saved bytes before writing the companion report.
    replay=verify_paired_candidate(
        bundle,problem=problem,m=m,k=k,tolerance=tolerance,
        expected_sha256=saved["artifact_sha256"],
    )
    if (replay["independent_objective_alignment"]!="pass" or
            replay["expected_digest_match"] is not True):
        raise AssertionError("Immutable pair failed independent replay")
    output={
        **analyzed,
        "source_sha":_git_commit(),
        "python":sys.version.split()[0],
        "platform":platform.platform(),
        "numpy":_version("numpy"),
        "scipy":_version("scipy"),
        "production_sampler_n_points":n_points,
        "production_default_scalar_tol":1e-7,
        "deterministic_source_seed":42+m,
        "source_ray_pool_size":max(4*n_points,256),
        "source_gaussian_ray_count":max(4*n_points,256)//2,
        "source_gamma_ray_count":max(4*n_points,256)//4,
        "source_expected_total_scalarizations":(
            max(4*n_points,256)*7//4
        ),
        "decision_matrix_sha256":hashlib.sha256(ps.tobytes()).hexdigest(),
        "objective_matrix_sha256":hashlib.sha256(f.tobytes()).hexdigest(),
        "paired_artifact_sha256":saved["artifact_sha256"],
    }
    with Path(report).open("x",encoding="utf-8") as stream:
        json.dump(output,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem",choices=("DTLZ5","DTLZ6"),required=True)
    parser.add_argument("--m",type=int,default=4)
    parser.add_argument("--k",type=int,default=1)
    parser.add_argument("--points",type=int,default=EXPECTED_N)
    parser.add_argument("--tolerance",type=float,default=1e-9)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(argv)
    result=capture_and_audit(
        problem=args.problem,m=args.m,k=args.k,n_points=args.points,
        bundle=args.bundle,report=args.report,tolerance=args.tolerance,
    )
    print(json.dumps({
        "problem":args.problem,
        "n_points":result["n_points"],
        "radial":result["radial"],
        "theorems":result["exact_rational_g_decision_image_ledger"]["counts"],
        "irrational_support":(
            result["dtlz6_irrational_g_support_ledger"]["counts"]
            if result["dtlz6_irrational_g_support_ledger"] else None),
        "geometry":result["geometry_observational"],
        "source_sha":result["source_sha"],
        "artifact_sha256":result["paired_artifact_sha256"],
        "scientific_certification":"unverified",
    },sort_keys=True,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
