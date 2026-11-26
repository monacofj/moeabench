# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only multi-resolution *rigorous* patch-to-actual-PS cover.

Repeat the already proved DTLZ5/6 continuous restricted positive-g
patch-to-actual selected PS certificates at two exactly nested dyadic grids.
Report BOTH rigorous bounds and their minimum (also rigorous). No
inference about the full continuous PF, pointwise PS membership or
indicator quality. Paired-source bytes and proof records are retained.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .dtlz5_patch_actual_source_cover import (
    certify_dtlz5_patch_to_source_cover,
    replay_dtlz5_patch_to_source_cover,
    _decode,
)
from .dtlz6_patch_actual_source_cover import (
    certify_dtlz6_patch_to_source_cover,
    replay_dtlz6_patch_to_source_cover,
)

GRIDS = {
    "DTLZ5": (
        {"x1_grid":(0.,1/16,1/8),
         "tail_grid":(.5,5/8,3/4)},
        {"x1_grid":(0.,1/32,1/16,3/32,1/8),
         "tail_grid":(.5,9/16,5/8,11/16,3/4)},
    ),
    "DTLZ6": (
        {"x1_grid":(0.,1/16,1/8),
         "u_grid":(0.,1/8,1/4)},
        {"x1_grid":(0.,1/32,1/16,3/32,1/8),
         "u_grid":(0.,1/16,1/8,3/16,1/4)},
    ),
}


def multiresolution_source_certificate(decisions, *, problem):
    if problem not in GRIDS:
        raise ValueError("Only original DTLZ5 or DTLZ6 supported")
    cert,replay = (
        (certify_dtlz5_patch_to_source_cover,
         replay_dtlz5_patch_to_source_cover)
        if problem=="DTLZ5"
        else (certify_dtlz6_patch_to_source_cover,
              replay_dtlz6_patch_to_source_cover)
    )
    records=[]
    for i,grid in enumerate(GRIDS[problem]):
        proof=cert(decisions,**grid)
        if not replay(decisions,proof,**grid):
            raise ArithmeticError("A component covering certificate failed replay")
        bound=_decode(proof["continuous_patch_to_actual_source_l2_upper"])
        grid_term=_decode(proof["continuous_patch_to_grid_l2_upper"])
        source_term=_decode(proof["worst_grid_to_source_l1_upper"])
        if bound!=grid_term+source_term:
            raise ArithmeticError("Cover decomposition is inconsistent")
        records.append({
            "grid":("3x3" if i==0 else "5x5"),
            "configuration":{name:[float(x).hex() for x in axis]
                             for name,axis in grid.items()},
            "certificate":proof,
            "continuum_to_grid_l2_upper":_encode_local(grid_term),
            "grid_to_selected_source_l1_upper":_encode_local(source_term),
            "patch_to_selected_source_l2_upper":_encode_local(bound),
        })
    numbers=[
        _decode(r["patch_to_selected_source_l2_upper"]) for r in records
    ]
    best=min(range(len(numbers)),key=lambda i:(numbers[i],i))
    return {
        "schema":"dtlz56_m4k1_original_selected_patch_multiresolution_v1",
        "problem":problem,"M":4,"K":1,
        "conclusion":"rigorous_restricted_patch_multiresolution_bound",
        "records":records,
        "best_grid":records[best]["grid"],
        "best_rigorous_upper":_encode_local(numbers[best]),
        "actual_selected_source_only":True,
        "complete_positive_g_pf_coverage":"unverified",
        "indicator_fitness":"unverified",
        "scientific_certification":"unverified",
    }


def _encode_local(value):
    from .dtlz56_exact_same_ray import _encode
    return _encode(value)


def replay_multiresolution_source_certificate(decisions, record, *, problem):
    try:
        expected=multiresolution_source_certificate(
            decisions,problem=problem)
        return (record==expected and expected["conclusion"]==
                "rigorous_restricted_patch_multiresolution_bound")
    except (TypeError,ValueError,ArithmeticError,OverflowError,
            ZeroDivisionError):
        return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-bundle",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Research report overwrite prohibited")
    digest=hashlib.sha256(args.paired_bundle.read_bytes()).hexdigest()
    with np.load(args.paired_bundle,allow_pickle=False) as data:
        if set(data.files)!={"decisions","objectives","problem","M","K"}:
            raise ValueError("Unexpected paired artifact members")
        problem=str(data["problem"].item())
        if problem not in GRIDS or int(data["M"])!=4 or int(data["K"])!=1:
            raise ValueError("Unexpected original problem or dimension")
        decisions=data["decisions"].copy()
    report=multiresolution_source_certificate(decisions,problem=problem)
    if not replay_multiresolution_source_certificate(
        decisions,report,problem=problem):
        raise ArithmeticError("Independent research replay failed")
    report["paired_artifact_sha256"]=digest
    with args.report.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    for record in report["records"]:
        print(problem,record["grid"],
              "continuum-to-grid",float(_decode(
                  record["continuum_to_grid_l2_upper"])),
              "grid-to-actual",float(_decode(
                  record["grid_to_selected_source_l1_upper"])),
              "rigorous-total",float(_decode(
                  record["patch_to_selected_source_l2_upper"])))
    print(problem,"best rigorous restricted patch bound",
          float(_decode(report["best_rigorous_upper"])))
    print("paired artifact SHA256",digest)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
