#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read-only comparison of corrected DPF2/DTLZ7 PF sources with frozen GT.

This *does not* recalibrate, rewrite or certify an old reference. Archived
GT may include dominated points because the historical Pareto samplers
covered the entire unit-parameter cube, rather than the true record set.
The output is a bounded, reproducible observation; it is not a proof of
global-Pareto compliance of the frozen 10k archive.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from moeabench.mops import DPF2, DTLZ7

FROZEN=(ROOT/"moeabench/diagnostics/resources/baselines_v0.14.0.json").resolve()
PKG_DIR=(ROOT/"moeabench/diagnostics/resources/references").resolve()
RESOURCE_DIR=(ROOT/"moeabench/diagnostics/resources").resolve()
REPORTS=(ROOT/"calibration/reports").resolve()
OFFICIAL_REPORT=(ROOT/"calibration/audit_report.html").resolve()


def _sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(
        x,dtype=np.float64)).tobytes()).hexdigest()


def _nondominated_fraction(x):
    if x.ndim!=2 or len(x)<2 or not np.all(np.isfinite(x)):
        raise ValueError("Expected finite objective populations")
    unique=np.unique(x,axis=0)
    nd=NonDominatedSorting().do(unique,only_non_dominated_front=True)
    return {
        "unique_points":int(len(unique)),
        "nondominated_points":int(len(nd)),
        "nondominated_fraction_within_sample":float(len(nd)/len(unique)),
    }


def observe(name, *, n_points=1024, seed=1337):
    if name not in ("DPF2","DTLZ7"):
        raise ValueError("Only the two corrected benchmark PF sources are supported")
    source=json.loads(FROZEN.read_text())
    path=PKG_DIR/name/"calibration_package.npz"
    with np.load(path,allow_pickle=False) as pkg:
        gt=np.asarray(source["problems"][name]["gt_reference"],dtype=np.float64)
        pkg_gt=np.asarray(pkg["gt_norm"],dtype=np.float64)
        ideal=np.asarray(pkg["ideal"],dtype=np.float64)
        nadir=np.asarray(pkg["nadir"],dtype=np.float64)
    if not np.array_equal(gt,pkg_gt):
        raise ValueError(f"{name}: frozen JSON and NPZ ground truth disagree")
    if not (gt.ndim==2 and len(gt)>=n_points and n_points>=64
            and np.all(nadir>ideal)):
        raise ValueError("Invalid bounded legacy sample or normalization")
    rng=np.random.default_rng(seed)
    selection=rng.choice(len(gt),size=n_points,replace=False)
    archived=gt[selection]
    mop={"DPF2":DPF2,"DTLZ7":DTLZ7}[name]()
    corrected_raw=np.asarray(mop.pf(n_points=n_points),dtype=np.float64)
    corrected=(corrected_raw-ideal)/(nadir-ideal)
    if not np.all(np.isfinite(corrected)):
        raise ValueError("Analytical source has nonfinite objectives")
    gt_tree=cKDTree(gt)
    spacing=float(np.median(gt_tree.query(gt,k=2)[0][:,1]))
    if spacing<=0 or not np.isfinite(spacing):
        raise ValueError("No positive GT local spacing")
    nn=gt_tree.query(corrected,k=1)[0]
    return {
        "problem":name,
        "status":"NONCERTIFIED_FROZEN_GT_COMPATIBILITY_OBSERVATION",
        "archived_baseline_version":source["version"],
        "frozen_gt_sha256":_sha(gt),
        "frozen_GT_points":int(len(gt)),
        "legacy_subsample_seed":int(seed),
        "legacy_subsample_points":int(n_points),
        "legacy_within_sample":_nondominated_fraction(archived),
        "corrected_pf":_nondominated_fraction(corrected),
        "corrected_pf_sha256":_sha(corrected),
        "corrected_to_frozen_GT_nearest_spacing_units":{
            "median":float(np.median(nn)/spacing),
            "p95":float(np.percentile(nn,95)/spacing),
            "max":float(np.max(nn)/spacing),
        },
        "frozen_GT_must_not_be_assumed_recaptured_by_new_pf":True,
        "certified":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points",type=int,default=1024)
    p.add_argument("--seed",type=int,default=1337)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() or out.is_relative_to(RESOURCE_DIR)
            or out.is_relative_to(REPORTS) or out==OFFICIAL_REPORT):
        p.error("Refusing existing or official scientific destination")
    report={
        "status":"NONCERTIFIED_ARCHIVED_PF_GT_SOURCE_COMPATIBILITY_AUDIT",
        "problems":{k:observe(k,n_points=a.points,seed=a.seed)
                    for k in ("DPF2","DTLZ7")},
        "baseline_sha256":hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "no_official_files_changed":True,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        name:{
            "legacy_within_sample":v["legacy_within_sample"],
            "corrected_pf":v["corrected_pf"],
            "nearest_GT":v["corrected_to_frozen_GT_nearest_spacing_units"],
        } for name,v in report["problems"].items()
    },indent=2))


if __name__=="__main__":
    main()
