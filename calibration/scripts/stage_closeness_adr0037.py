#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage ONLY ADR 0037 finite-GT Closeness residue corrections.

The frozen calibration's original ideal_res was calculated from a subset
of the SAME GT, making it zero by construction. The existing MOP analytical
PF generator is used here to obtain a fresh, GT-nonoverlapping reference,
without imposing an unapproved uniform-objective-area sampling law.

This script never modifies baselines.json, a versioned source baseline,
the canonical audit_report, or any GT. Output is NONCERTIFIED until
independent scientific acceptance of each MOP/K reference and Q behavior.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moeabench.diagnostics import baselines
from moeabench.diagnostics.calibration import estimate_ideal_residue
from calibration.scripts.generate_baselines import _analytical_ideal_from_package

FROZEN = ROOT / "moeabench/diagnostics/resources/baselines_v0.14.0.json"
RESOURCES = (ROOT / "moeabench/diagnostics/resources").resolve()
REPORTS = (ROOT / "calibration/reports").resolve()
CANONICAL_REPORT = (ROOT / "calibration/audit_report.html").resolve()

# The 14-MOP science audit (docs/fair_pf_source_audit_issue75.md)
# refuted the near-GT-only assumption for these MOPs. DPF2 and DTLZ7
# have *now* corrected source samplers, but their frozen v0.14.0 GT contains
# dominated historical objectives. DPF5's collapsed branch introduces an
# undefined probability weight after objective deduplication.
#
# A runtime source fix is not a license to stage new Q anchors against an
# invalid/mixed-measure historical GT. These explicit holds are about THIS
# archive, not a permanent ban on a future independently reviewed version.
HELD_ARCHIVAL_SOURCES = {
    "DPF2": "Historical GT includes dominated points; corrected PF cannot certify frozen reference",
    "DTLZ7": "Historical GT includes dominated points; corrected PF cannot certify frozen reference",
    "DPF5": "Mixed-dimensional PF branch weights change under objective deduplication",
}


def revise_closeness_record(old, epsilon):
    """Return a copy correcting the physical metric and its null ECDF together."""
    if "ideal_res_method" in old or abs(float(old.get("ideal_res", 0.0))) > 1e-14:
        raise ValueError("Refusing double correction or unknown residue provenance")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("A fresh ideal PF must yield a positive finite residue")
    ecdf = np.asarray(old["rand_ecdf"], dtype=float)
    if (ecdf.ndim != 1 or not len(ecdf) or not np.all(np.isfinite(ecdf))
            or np.any(np.diff(ecdf) < 0)):
        raise ValueError("Invalid frozen Closeness ECDF")
    updated = dict(old)
    updated["ideal_res"] = float(epsilon)
    updated["ideal_res_method"] = "independent_analytical_pf"
    updated["rand_ecdf"] = np.maximum(0.0, ecdf - epsilon).tolist()
    updated["rand50"] = max(0.0, float(old["rand50"]) - epsilon)
    if "p95" in old:
        updated["p95"] = max(0.0, float(old["p95"]) - epsilon)
    return updated


def stage(source, problems, *, packages=RESOURCES / "references"):
    """Return (noncertified candidate, per-MOP/K impact); no write side effects."""
    if not problems or len(problems) != len(set(problems)):
        raise ValueError("Require distinct requested MOP names")
    candidate = json.loads(json.dumps(source))
    report = {
        "status": "PROVISIONAL_ADR0037_INDEPENDENT_MOP_PF_NOT_CERTIFIED",
        "method": "existing_mop_analytical_pf_independent_gt",
        "objective_area_uniform": False,
        "source_calibration_version": source.get("version"),
        "source_version_retained": True,
        "supported": {},
        "unsupported": {},
    }
    for name in problems:
        if name in HELD_ARCHIVAL_SOURCES:
            report["unsupported"][name] = "HOLD_ARCHIVAL_REFERENCE: " + HELD_ARCHIVAL_SOURCES[name]
            continue
        records = source.get("problems", {}).get(name)
        pkgpath = Path(packages) / name / "calibration_package.npz"
        if not isinstance(records, dict) or not pkgpath.is_file():
            report["unsupported"][name] = "Missing original problem or packaged GT"
            continue
        with np.load(pkgpath, allow_pickle=False) as pkg:
            gt = np.asarray(records["gt_reference"], dtype=np.float64)
            frozen = np.asarray(pkg["gt_norm"], dtype=np.float64)
            if gt.ndim != 2 or not np.array_equal(gt, frozen):
                report["unsupported"][name] = "Source/packaged GT differ; cannot share calibration"
                continue
            ideal = _analytical_ideal_from_package(name, gt, pkg)
        if ideal is None:
            report["unsupported"][name] = "No compatible independent analytical PF"
            continue
        changed = {}
        for key, krec in records.items():
            if not key.isdigit() or "closeness" not in krec:
                continue
            K = int(key)
            scale = baselines.get_resolution_factor_k(gt, K, seed=0)
            eps = estimate_ideal_residue(gt, ideal, scale)
            before = krec["closeness"]
            revised = revise_closeness_record(before, eps)
            candidate["problems"][name][key]["closeness"] = revised
            changed[key] = {
                "s_K": float(scale),
                "epsilon": float(eps),
                "rand50_before": float(before["rand50"]),
                "rand50_after": float(revised["rand50"]),
            }
        if not changed:
            report["unsupported"][name] = "No eligible Closeness K records"
            continue
        report["supported"][name] = {
            "gt_sha256": hashlib.sha256(np.ascontiguousarray(gt).tobytes()).hexdigest(),
            "gt_points": len(gt),
            "independent_pf_points": len(ideal),
            "pf_law": "existing_mop_pf_sampling_distribution",
            "K": changed,
        }
    report["K_records_changed"] = sum(len(x["K"]) for x in report["supported"].values())
    report["MOPs_untouched"] = len(source.get("problems", {})) - len(report["supported"])
    return candidate, report


def verify(source, candidate, report):
    """Independent replay of changed Closeness-only ECDFs; fail closed."""
    if report.get("status") != "PROVISIONAL_ADR0037_INDEPENDENT_MOP_PF_NOT_CERTIFIED":
        raise ValueError("Invalid science status")
    if source.keys() != candidate.keys() or any(
        source[k] != candidate[k] for k in source if k != "problems"
    ):
        raise ValueError("Source metadata changed")
    if set(source["problems"]) != set(candidate["problems"]):
        raise ValueError("MOP inventory changed")
    covered = set(report["supported"])
    if covered.intersection(HELD_ARCHIVAL_SOURCES):
        raise ValueError("Cannot certify held historical GT/branch measure in Closeness migration")
    if any(name in source["problems"] and name not in report["unsupported"]
           for name in HELD_ARCHIVAL_SOURCES):
        raise ValueError("Held historical source missing explicit unsupported status")
    for name, records in source["problems"].items():
        revised = candidate["problems"][name]
        if set(records) != set(revised):
            raise ValueError(f"MOP/K inventory changed: {name}")
        if name not in covered:
            if records != revised:
                raise ValueError(f"Unsupported MOP was modified: {name}")
            continue
        for key, before in records.items():
            after = revised[key]
            if not key.isdigit() or "closeness" not in before:
                if before != after:
                    raise ValueError(f"Non-Closeness metadata changed: {name}/{key}")
                continue
            if set(before) != set(after):
                raise ValueError(f"Unrelated metric inventory changed: {name}/{key}")
            for metric in before:
                if metric != "closeness" and before[metric] != after[metric]:
                    raise ValueError(f"Unrelated FAIR metric changed: {name}/{key}/{metric}")
            old = before["closeness"]
            new = after["closeness"]
            expected = report["supported"][name]["K"][key]["epsilon"]
            if (new.get("ideal_res_method") != "independent_analytical_pf"
                    or not np.isclose(float(new["ideal_res"]), expected, atol=1e-12, rtol=0)):
                raise ValueError(f"Missing analytical provenance: {name}/{key}")
            arrays = (
                np.asarray(old["rand_ecdf"], dtype=float),
                np.asarray(new["rand_ecdf"], dtype=float),
            )
            np.testing.assert_allclose(arrays[1], np.maximum(0, arrays[0] - expected),
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(new["rand50"],
                                       max(0, float(old["rand50"]) - expected),
                                       atol=1e-12, rtol=0)
            if "p95" in old:
                np.testing.assert_allclose(new["p95"],
                                           max(0, float(old["p95"]) - expected),
                                           atol=1e-12, rtol=0)
            allowed = {"ideal_res", "rand_ecdf", "rand50", "p95", "ideal_res_method"}
            if (set(new) - set(old)) != {"ideal_res_method"}:
                raise ValueError(f"Unexpected Closeness fields: {name}/{key}")
            if any(old[k] != new.get(k) for k in old if k not in allowed):
                raise ValueError(f"Unrelated Closeness field changed: {name}/{key}")
    return {"status": "PASS_PROVISIONAL_ONLY",
            "MOPs_changed": sorted(covered),
            "K_records_checked": report["K_records_changed"],
            "official_files_modified": False}


def _safe_output(path, inputs):
    destination = Path(path).resolve()
    if (destination.exists() or destination in inputs
            or destination.is_relative_to(RESOURCES)
            or destination.is_relative_to(REPORTS)
            or destination == CANONICAL_REPORT):
        raise ValueError("Refusing existing or official-baseline/report output")
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=FROZEN)
    p.add_argument("--problem", action="append", dest="problems")
    p.add_argument("--candidate", type=Path)
    p.add_argument("--impact", type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.input.read_text())
    requested = a.problems or sorted(source["problems"])
    candidate, report = stage(source, requested)
    report["source_sha256"] = hashlib.sha256(a.input.read_bytes()).hexdigest()
    validation = verify(source, candidate, report)
    report["verification"] = validation
    report["candidate_sha256"] = hashlib.sha256(
        json.dumps(candidate,sort_keys=True).encode()
    ).hexdigest()
    protected = {a.input.resolve()}
    output_impact = _safe_output(a.impact, protected)
    output_candidate = _safe_output(a.candidate, protected) if a.candidate else None
    if output_candidate is not None and output_impact == output_candidate:
        p.error("Impact and candidate outputs must be separate")
    output_impact.parent.mkdir(parents=True, exist_ok=True)
    output_impact.write_text(json.dumps(report,indent=2)+"\n")
    if output_candidate:
        output_candidate.parent.mkdir(parents=True,exist_ok=True)
        output_candidate.write_text(json.dumps(candidate,separators=(",",":")))
    print(json.dumps({
        "status":report["status"],
        "supported":sorted(report["supported"]),
        "unsupported":report["unsupported"],
        "K_records_changed":report["K_records_changed"],
        "verification":validation,
    },indent=2))


if __name__ == "__main__":
    main()
