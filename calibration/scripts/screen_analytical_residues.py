#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Inventory all frozen calibration GTs for independent Closeness residues.

A screening study, NOT baseline certification or a substitute for full
recalibration. In particular, agreement of the analytical MOP sampler with
the GT at its local nearest-neighbor resolution is a screening criterion,
not a proof that the two analytical front distributions coincide.

No historical JSON is modified. Unsupported problems remain unsupported,
with their explicit error/provenance rather than an invented correction.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree

from moeabench.diagnostics import baselines
from moeabench.diagnostics.calibration import estimate_ideal_residue
from calibration.scripts.generate_baselines import _analytical_ideal_from_package


def inventory(source, *, k_values=(20, 100), root=ROOT):
    report = {"status": "NONCERTIFIED_ANALYTICAL_COMPATIBILITY_SCREEN",
              "k_values": list(k_values), "supported": {}, "unsupported": {}}
    references = root/"moeabench/diagnostics/resources/references"
    for name in sorted(source["problems"]):
        t0 = time.perf_counter()
        entry = source["problems"][name]
        pkgpath = references/name/"calibration_package.npz"
        try:
            if not pkgpath.is_file():
                raise ValueError("No frozen calibration package")
            gt = np.asarray(entry["gt_reference"], dtype=np.float64)
            with np.load(pkgpath, allow_pickle=False) as pkg:
                frozen = np.asarray(pkg["gt_norm"], dtype=np.float64)
                if not np.array_equal(gt, frozen):
                    raise ValueError("Baseline GT does not match packaged GT")
                ideal = _analytical_ideal_from_package(name, gt, pkg)
            if ideal is None:
                raise ValueError("No independently verified analytical PF for archived GT")
            if len(ideal) < 2 or not np.all(np.isfinite(ideal)):
                raise ValueError("No finite independent ideal sample")
            nearest = cKDTree(gt).query(ideal, k=1)[0]
            if np.any(nearest <= 1e-12):
                raise ValueError("Independent analytical sample overlaps archived GT")
            scales = {}
            for k in k_values:
                if k < 2 or k > len(gt):
                    raise ValueError(f"Invalid K={k} for GT size {len(gt)}")
                s_k = baselines.get_resolution_factor_k(gt, k, seed=0)
                epsilon = estimate_ideal_residue(gt, ideal, s_k)
                if not np.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError(f"Nonpositive residue for K={k}")
                scales[str(k)] = {"s_k": float(s_k), "ideal_res": float(epsilon)}
            report["supported"][name] = {
                "shape": list(gt.shape),
                "fresh_points": int(len(ideal)),
                "gt_sha256": hashlib.sha256(np.ascontiguousarray(gt).tobytes()).hexdigest(),
                "method": "independent_analytical_pf_screened",
                "k": scales,
                "elapsed_s": time.perf_counter()-t0,
            }
        except (ValueError, KeyError, TypeError, AttributeError,
                OSError, IndexError) as exc:
            report["unsupported"][name] = {
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsed_s": time.perf_counter()-t0,
            }
    report["total"] = len(source["problems"])
    report["supported_count"] = len(report["supported"])
    report["unsupported_count"] = len(report["unsupported"])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"moeabench/diagnostics/resources/baselines_v0.14.0.json")
    parser.add_argument("--k",type=int,action="append",dest="k_values")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():
        parser.error("Refusing to overwrite previous screening report")
    if out.is_relative_to((ROOT/"moeabench/diagnostics/resources").resolve()) or out.is_relative_to(
        (ROOT/"calibration/reports").resolve()
    ):
        parser.error("Screening must not be published in certified artifact directories")
    with args.input.open(encoding="utf-8") as fh:
        source=json.load(fh)
    result=inventory(source,k_values=tuple(args.k_values or (20,100)))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("supported","unsupported")},indent=2))
    for name,rec in result["supported"].items():
        print("SUPPORTED",name,"K:",rec["k"],"elapsed_s:",round(rec["elapsed_s"],3),flush=True)
    for name,rec in result["unsupported"].items():
        print("UNSUPPORTED",name,rec["reason"],flush=True)
    if not result["supported"]:
        raise SystemExit("No compatible analytical reference; certification blocked")


if __name__=="__main__":
    main()
