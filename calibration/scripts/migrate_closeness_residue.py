#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preview or stage ADR-0037 finite-GT residue fixes without recalibrating ECDFs.

The frozen v0.14.0 closeness baselines were computed with a zero residue by
construction (FPS from the *same* GT). Once an independent analytical sample
establishes epsilon, the physical statistic transforms each sample as
u' = max(0, u - epsilon). Apply that same deterministic mapping to the stored
blurred-closeness ECDF and its percentile anchors. Other FAIR metrics and the
historical baseline file remain untouched.

This tool is a release migration aid, not part of the runtime library.
By default it performs a dry-run and prints a small impact report. Writing a
new full baseline requires an explicit --output path distinct from --input.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moeabench.diagnostics import baselines
from moeabench.diagnostics.calibration import estimate_ideal_residue
from calibration.scripts.generate_baselines import _analytical_ideal_from_package
from calibration.scripts.independent_pf_sources_issue75 import (
    normalized_independent_pf, sample_dtlz2_uniform_normalized_surface,
)

NORMALIZED_AREA_METHOD = "uniform_normalized_objective_pf_area_v1"
NORMALIZED_AREA_MEASURE = "uniform_intrinsic_area_normalized_objective_pf"


def residue_update_record(record, epsilon, *, method="independent_analytical_pf", provenance=None):
    """Return a residue-corrected copy of a legacy closeness baseline record.

    The input is required to be an uncorrected legacy profile. An already
    migrated profile must raise, not silently apply epsilon twice.
    """
    old = float(record.get("ideal_res", 0.0))
    if abs(old) > 1e-14 or "ideal_res_method" in record:
        raise ValueError("Baseline has already been corrected or has unknown provenance")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("invalid epsilon")
    out = dict(record)
    raw = np.asarray(record["rand_ecdf"], dtype=float)
    if raw.ndim != 1 or len(raw) == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("invalid raw closeness ECDF")
    out["rand_ecdf"] = np.maximum(0.0, raw - epsilon).tolist()
    out["rand50"] = max(0.0, float(record["rand50"]) - epsilon)
    if "p95" in out:
        out["p95"] = max(0.0, float(out["p95"]) - epsilon)
    out["ideal_res"] = float(epsilon)
    out["ideal_res_method"] = method
    if method == NORMALIZED_AREA_METHOD:
        if not isinstance(provenance, dict):
            raise ValueError("Normalized-area correction requires sampling provenance")
        out["reference_measure"] = NORMALIZED_AREA_MEASURE
        out["ideal_res_sampler"] = str(provenance["law"])
        out["ideal_res_sampling_seed"] = int(provenance["seed"])
        out["ideal_res_sample_count"] = int(provenance["n_fresh"])
    elif method != "independent_analytical_pf":
        raise ValueError("Unknown ideal residue correction method")
    return out


def preview_migration(source, problems, *, reference_law="legacy_independent_pf",
                      sample_size=4096, sampling_seed=123):
    """Return (modified baseline JSON, compact per-problem impact report).

    Reproducing the *exact* packaged GT is a prerequisite, as is a fresh
    analytical sample with no coincident GT points. Unsupported problems are
    reported, never assigned a synthetic positive residue.
    """
    if reference_law not in ("legacy_independent_pf", "normalized_objective_area"):
        raise ValueError("Unrecognized ideal-reference probability law")
    if not isinstance(sample_size, int) or sample_size < 2:
        raise ValueError("sample_size must be an integer >= 2")
    if not isinstance(sampling_seed, int) or sampling_seed < 0:
        raise ValueError("sampling_seed must be a nonnegative integer")
    modified = json.loads(json.dumps(source))
    report = {
        "method": "ADR0037-independent-PF",
        "reference_law":reference_law,
        "status":"NONCERTIFIED_CANDIDATE",
        "problems":{}, "failed":{},
    }
    for problem in problems:
        if problem not in source.get("problems", {}):
            report["failed"][problem] = "problem absent from baseline"
            continue
        package_path = ROOT / "moeabench" / "diagnostics" / "resources" / "references" / problem / "calibration_package.npz"
        if not package_path.is_file():
            report["failed"][problem] = "calibration reference package absent"
            continue
        with np.load(package_path, allow_pickle=False) as package:
            gt = np.asarray(source["problems"][problem].get("gt_reference"), dtype=float)
            if gt.shape != package["gt_norm"].shape or not np.array_equal(gt, package["gt_norm"]):
                report["failed"][problem] = "baseline GT differs from frozen package; incompatible"
                continue
            provenance = None
            if reference_law == "normalized_objective_area":
                ideal = np.asarray(package["ideal"], dtype=float)
                nadir = np.asarray(package["nadir"], dtype=float)
                if problem == "DTLZ1" and gt.shape[1] == 3:
                    independent, metadata = normalized_independent_pf(
                        problem, n_points=sample_size, seed=sampling_seed,
                        ideal=ideal, nadir=nadir, gt=gt,
                    )
                    law = metadata["law"]
                elif problem == "DTLZ2" and gt.shape[1] == 3:
                    independent, law = sample_dtlz2_uniform_normalized_surface(
                        sample_size, sampling_seed, ideal=ideal, nadir=nadir,
                    )
                else:
                    report["failed"][problem] = (
                        "normalized-objective surface-area reference not justified "
                        "for this MOP/dimension"
                    )
                    continue
                provenance = {
                    "law":law, "seed":sampling_seed,
                    "n_fresh":len(independent),
                    "target_measure":NORMALIZED_AREA_MEASURE,
                    "ideal":ideal.tolist(),"nadir":nadir.tolist(),
                }
            else:
                independent = _analytical_ideal_from_package(problem, gt, package)

        if independent is None:
            report["failed"][problem] = "no verified independent analytical ideal front"
            continue

        updates = {}
        for key, data in source["problems"][problem].items():
            if not key.isdigit() or "closeness" not in data:
                continue
            K = int(key)
            scale = baselines.get_resolution_factor_k(gt, K, seed=0)
            epsilon = estimate_ideal_residue(gt, independent, scale)
            before = data["closeness"]
            after = residue_update_record(
                before, epsilon,
                method=(NORMALIZED_AREA_METHOD
                        if reference_law == "normalized_objective_area"
                        else "independent_analytical_pf"),
                provenance=provenance,
            )
            modified["problems"][problem][key]["closeness"] = after
            updates[key] = {
                "s_k": float(scale),
                "old_ideal_res": float(before.get("ideal_res", 0.0)),
                "new_ideal_res": float(epsilon),
                "old_rand50": float(before["rand50"]),
                "new_rand50": float(after["rand50"]),
            }
        report["problems"][problem] = {
            "gt_points": len(gt),
            "independent_ideal_points": len(independent),
            "gt_sha256": hashlib.sha256(np.ascontiguousarray(gt).tobytes()).hexdigest(),
            "ideal_sample_provenance":provenance,
            "K": updates,
        }
    return modified, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "moeabench/diagnostics/resources/baselines_v0.14.0.json")
    parser.add_argument("--problem", action="append", required=True, help="MOP name; repeat for multiple problems")
    parser.add_argument("--reference-law",
                        choices=("legacy_independent_pf","normalized_objective_area"),
                        default="legacy_independent_pf",
                        help="Experimental normalized area applies only to verified DTLZ1/2 M3")
    parser.add_argument("--sample-size",type=int,default=4096)
    parser.add_argument("--sampling-seed",type=int,default=123)
    parser.add_argument("--output", type=Path, help="Explicit path for staged new full baseline; never overwrites input")
    parser.add_argument("--impact-report", type=Path, help="Path for lightweight per-K numerical diff")
    opts = parser.parse_args()
    with opts.input.open(encoding="utf-8") as fh:
        data = json.load(fh)

    modified, report = preview_migration(
        data, opts.problem, reference_law=opts.reference_law,
        sample_size=opts.sample_size, sampling_seed=opts.sampling_seed,
    )
    print(json.dumps(report, indent=2))
    if opts.impact_report:
        opts.impact_report.parent.mkdir(parents=True, exist_ok=True)
        opts.impact_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if opts.output:
        out = opts.output.resolve()
        protected = (ROOT/"moeabench/diagnostics/resources").resolve()
        if (out == opts.input.resolve() or out.exists()
                or out.is_relative_to(protected)
                or out.is_relative_to((ROOT/"calibration/reports").resolve())
                or out == (ROOT/"calibration/audit_report.html").resolve()):
            parser.error("Refusing overwrite or destination among frozen/certified artifacts")
        if report["failed"]:
            parser.error("Refusing to write partial baseline with failed requested problems")
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(json.dumps(modified, separators=(",", ":")), encoding="utf-8")
        print(f"Staged new baseline: {opts.output}")


if __name__ == "__main__":
    main()
