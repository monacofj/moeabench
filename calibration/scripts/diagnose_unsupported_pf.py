#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Diagnose independent analytical PF limitations for unsupported MOPs.

An exploratory *measurement*, not a calibration algorithm, baseline mutation,
or permission to waive reference-manifold compatibility. Records whether
today's analytical sampler reproduces frozen GT geometry, returns the
requested cardinality and yields new non-overlapping front points.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree

from moeabench import mops


def diagnose(name, root=ROOT, fresh_points=1024):
    path=root/"moeabench/diagnostics/resources/references"/name/"calibration_package.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    started=time.perf_counter()
    with np.load(path,allow_pickle=False) as data:
        gt=np.asarray(data["gt_norm"],dtype=float)
        ideal=np.asarray(data["ideal"],dtype=float)
        nadir=np.asarray(data["nadir"],dtype=float)
    if gt.ndim!=2 or len(gt)<3 or gt.shape[1]!=len(ideal) or gt.shape[1]!=len(nadir):
        raise ValueError("Incompatible frozen calibration arrays")
    span=nadir-ideal
    if np.any(span<=0):
        raise ValueError("Invalid calibration normalization span")
    sampler=getattr(mops,name)()
    tree=cKDTree(gt)
    spacing=float(np.median(tree.query(gt,k=2)[0][:,1]))
    if spacing<=0:
        raise ValueError("Degenerate archived GT nearest-neighbor spacing")

    result={
        "mop":name, "archived_shape":list(gt.shape),
        "archived_GT_nn_median":spacing,
        "mop_class":type(sampler).__name__,
        "sampling_probes":{},
    }
    for requested in (len(gt),len(gt)+fresh_points):
        key="matched_requested_count" if requested==len(gt) else "fresh_extension"
        before=time.perf_counter()
        try:
            current=np.asarray(sampler.pf(n_points=requested),dtype=float)
            if current.ndim!=2 or current.shape[1]!=gt.shape[1] or not np.all(np.isfinite(current)):
                raise ValueError(f"Invalid output shape or values {current.shape}")
            normalized=(current-ideal)/span
            distances=tree.query(normalized,k=1)[0]
            overlaps=int(np.count_nonzero(distances<=1e-10))
            result["sampling_probes"][key]={
                "requested":int(requested),
                "returned":int(len(normalized)),
                "returned_exact_requested":bool(len(normalized)==requested),
                "GT_overlap_count":overlaps,
                "fresh_point_count":int(len(normalized)-overlaps),
                "median_distance_to_GT":float(np.median(distances)),
                "median_distance_over_GT_NN_spacing":float(np.median(distances)/spacing),
                "p95_distance_over_GT_NN_spacing":float(np.percentile(distances,95)/spacing),
                "elapsed_s":time.perf_counter()-before,
            }
        except (ValueError,TypeError,AttributeError,RuntimeError,IndexError) as exc:
            result["sampling_probes"][key]={
                "requested":int(requested),
                "error":f"{type(exc).__name__}: {exc}",
                "elapsed_s":time.perf_counter()-before,
            }
    result["elapsed_total_s"]=time.perf_counter()-started
    result["status"]="DIAGNOSTIC_ONLY_NO_MOP_CALIBRATION_CERTIFIED"
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem",action="append",dest="problems",required=True)
    p.add_argument("--fresh-points",type=int,default=1024)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=args.output.resolve()
    if out.exists():
        p.error("Refusing to overwrite diagnostic results")
    if out.is_relative_to((ROOT/"calibration/reports").resolve()) or out.is_relative_to(
        (ROOT/"moeabench/diagnostics/resources").resolve()
    ):
        p.error("Diagnostics cannot be written to certified artifact directories")
    if args.fresh_points<2:
        p.error("Require at least 2 independent candidate samples")

    records={}
    for name in args.problems:
        try:
            records[name]=diagnose(name,fresh_points=args.fresh_points)
        except (ValueError,TypeError,AttributeError,RuntimeError,OSError,IndexError) as exc:
            records[name]={"status":"UNSUPPORTED_IN_DIAGNOSTIC",
                            "error":f"{type(exc).__name__}: {exc}"}
        print(json.dumps(records[name],indent=2),flush=True)

    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({
        "status":"NONCERTIFIED_SAMPLER_DIAGNOSIS",
        "no_frozen_baselines_modified":True,
        "mops":records,
    },indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
