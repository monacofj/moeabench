#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit how DPF5's frozen GT encodes the mixed-dimensional branch measure.

Non-destructive, bounded numerical read of the original baseline+NPZ.
Detect exact/nearest reproduction by the *actual* DPF5.pf(n_points=N)
generator, source multiplicity, on-sphere consistency and branch
histograms. Do not infer the historical production script from visual
geometry alone; observed correspondence is evidence, not authorship.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from moeabench.mops import DPF5
from calibration.scripts.study_dpf5_branch_measure_issue78 import (
    FROZEN, PACKAGE, RESOURCES, REPORTS, OFFICIAL, _sha,
)

def classify_branch(physical, tol=1e-9):
    """Conservative geometric branch detection using objective diagonal.

    For the D branch f1=f2 exactly; P has f1=f2 only on the x2=0.5
    parameter meridian. Boundary ambiguity is recorded separately.
    """
    physical=np.asarray(physical,dtype=np.float64)
    return np.abs(physical[:,0]-physical[:,1])<=tol


def _unique_nd_info(points):
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    u=np.unique(points,axis=0)
    nd=NonDominatedSorting().do(u,only_non_dominated_front=True)
    return {"unique":int(len(u)),"duplicate_rows":int(len(points)-len(u)),
            "nondominated_unique":int(len(nd)),
            "nondominated_fraction":float(len(nd)/len(u))}


def audit():
    frozen=json.loads(FROZEN.read_text())
    gt=np.asarray(frozen["problems"]["DPF5"]["gt_reference"],dtype=np.float64)
    with np.load(PACKAGE,allow_pickle=False) as pkg:
        packaged=np.asarray(pkg["gt_norm"],dtype=np.float64)
        ideal=np.asarray(pkg["ideal"],dtype=np.float64)
        nadir=np.asarray(pkg["nadir"],dtype=np.float64)
    if not np.array_equal(gt,packaged) or not np.all(nadir>ideal):
        raise ValueError("Frozen JSON GT differs from independent NPZ package")
    if gt.shape[1]!=3 or len(gt)<100:
        raise ValueError("Expected DPF5 M3 archived reference")
    model=DPF5(D=2,M=3)
    decisions=model.ps(n_points=len(gt))
    source=np.asarray(model.evaluation(decisions)["F"],dtype=np.float64)
    gen_norm=(source-ideal)/(nadir-ideal)
    if not np.all(np.isfinite(gen_norm)):
        raise ValueError("Invalid source")
    expected_count=int(len(gt))
    original_physical=gt*(nadir-ideal)+ideal
    radius=np.linalg.norm(original_physical,axis=1)
    source_radius=np.linalg.norm(source,axis=1)
    frozen_info=_unique_nd_info(gt)
    source_info=_unique_nd_info(gen_norm)

    # Row-wise equality is stronger than mere two-way Hausdorff closeness.
    # Sort lexicographically as the archived GT need not preserve source row order.
    def sorted_rows(x):
        return x[np.lexsort(tuple(x[:,i] for i in reversed(range(x.shape[1]))))]
    same_size=len(gen_norm)==len(gt)
    source_tree=cKDTree(gen_norm)
    gt_tree=cKDTree(gt)
    dist_gt_to_source=source_tree.query(gt,k=1)[0]
    dist_source_to_gt=gt_tree.query(gen_norm,k=1)[0]
    gt_diag=classify_branch(original_physical)
    src_diag=classify_branch(source)
    src_exact_D=decisions[:,0]>1/3
    # Categorize actual D as certain source D, and source P on the
    # diagonal separately (ambiguous if only objectives were observed).
    source_D_count=int(np.count_nonzero(src_exact_D))
    source_P_diagonal=int(np.count_nonzero(src_diag&~src_exact_D))
    frozen_diag_count=int(np.count_nonzero(gt_diag))
    x1_orig=2*np.arcsin(np.clip(original_physical[:,2],0,1))/np.pi
    x1_src=decisions[:,0]
    hist_edges=np.linspace(0,1,13)
    counts_old=np.histogram(x1_orig,bins=hist_edges)[0]
    counts_src=np.histogram(x1_src,bins=hist_edges)[0]
    empirical_tv=.5*np.sum(np.abs(counts_old/len(gt)-counts_src/len(decisions)))

    return {
        "status":"NONCERTIFIED_DPF5_FROZEN_GT_DENSITY_PROVENANCE_AUDIT",
        "no_official_files_changed":True,
        "reference_baseline_version":frozen.get("version"),
        "frozen_GT_sha256":_sha(gt),
        "archived_N":expected_count,
        "direct_mop_ps_generated_N":int(len(gen_norm)),
        "current_mop_pf_objective_sha256":_sha(gen_norm),
        "archives_match_generator_rowwise":bool(
            same_size and np.allclose(gen_norm,gt,atol=1e-12,rtol=0)),
        "archives_match_generator_sorted_rows":bool(
            same_size and np.allclose(
                sorted_rows(gen_norm),sorted_rows(gt),atol=1e-12,rtol=0)),
        "nearest_gt_to_generated":{
            "median":float(np.median(dist_gt_to_source)),
            "p95":float(np.percentile(dist_gt_to_source,95)),
            "max":float(np.max(dist_gt_to_source)),
            "fraction_within_1e-9":float(np.mean(dist_gt_to_source<1e-9)),
        },
        "nearest_generated_to_gt":{
            "median":float(np.median(dist_source_to_gt)),
            "p95":float(np.percentile(dist_source_to_gt,95)),
            "max":float(np.max(dist_source_to_gt)),
            "fraction_within_1e-9":float(np.mean(dist_source_to_gt<1e-9)),
        },
        "archived_unique_and_nondominated":frozen_info,
        "current_sampler_unique_and_nondominated":source_info,
        "archived_spherical_radial_max_error":float(np.max(np.abs(radius-1))),
        "current_sampler_radial_max_error":float(np.max(np.abs(source_radius-1))),
        "archived_on_diagonal_count":frozen_diag_count,
        "archived_on_diagonal_fraction":float(np.mean(gt_diag)),
        "current_generator_true_D_count":source_D_count,
        "current_generator_true_D_fraction":float(np.mean(src_exact_D)),
        "current_generator_P_ambiguous_diagonal_count":source_P_diagonal,
        "current_generator_diagonal_count":int(np.count_nonzero(src_diag)),
        "x1_bins":hist_edges.tolist(),
        "x1_archived_histogram":counts_old.tolist(),
        "x1_generated_histogram":counts_src.tolist(),
        "x1_histogram_total_variation":float(empirical_tv),
        "GT_can_be_read_as_uniform_decision_grid_pushforward":bool(
            same_size and np.allclose(
                sorted_rows(gen_norm),sorted_rows(gt),atol=1e-12,rtol=0)),
        "certified":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() or out.is_relative_to(RESOURCES)
            or out.is_relative_to(REPORTS) or out==OFFICIAL):
        p.error("Refusing existing/official scientific destination")
    result=audit()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({
        "N":result["archived_N"],
        "match_sorted":result["archives_match_generator_sorted_rows"],
        "nearest":result["nearest_gt_to_generated"],
        "archived_nd":result["archived_unique_and_nondominated"],
        "current_nd":result["current_sampler_unique_and_nondominated"],
        "archived_diagonal_mass":result["archived_on_diagonal_fraction"],
        "expected_true_D_mass":result["current_generator_true_D_fraction"],
        "x1_histogram_tv":result["x1_histogram_total_variation"],
    },indent=2))


if __name__=="__main__":
    main()
