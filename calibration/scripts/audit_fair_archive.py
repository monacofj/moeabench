#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Provisional FAIR-only audit from archived fronts (issue #75 / ADR 0051).

Archived CSV files contain final fronts and GTs but generally not generation
zero. Consequently physical HEADWAY cannot be reconstructed for those runs.
Neither the historical experimental GTs nor their coordinates have been
certified equivalent to the bundled clinical calibration GT, so this script
NEVER generates clinical Q-scores.

This produces a *separate provisional JSON*, NOT the certified audit_v*.json
schema consumed by the published HTML generator. The script refuses to write
under calibration/reports or to overwrite an existing file.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from moeabench.diagnostics import baselines, fair

try:
    from .audit_report_preflight import RUN_PATTERN, ROOT
except ImportError:
    from audit_report_preflight import RUN_PATTERN, ROOT

METRIC_NAMES = ("CLOSENESS", "COVERAGE", "GAP", "REGULARITY", "BALANCE", "HEADWAY")


def _read_front(path: Path, *, header: bool) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",", skip_header=1 if header else 0)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or len(values) < 2 or values.shape[1] != 3:
        raise ValueError(f"{path}: expected >=2 rows and 3 objective columns")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: nonfinite objective coordinates")
    return values


def _nondominated(points: np.ndarray) -> np.ndarray:
    """Independent Pareto-minimization oracle, not a MoeaBench metric."""
    # For each point i, reject it if another point dominates it (including
    # strict improvement on at least one objective). Preserve original order.
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        dominated = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        if np.any(dominated):
            keep[i] = False
    return points[keep]


def compute_fair_only(final: np.ndarray, gt: np.ndarray, initial=None) -> dict:
    """Compute physical metrics only, with no clinical baseline borrowing."""
    final = _nondominated(np.asarray(final, dtype=float))
    gt = np.asarray(gt, dtype=float)
    if len(final) < 2:
        raise ValueError("Fewer than two non-dominated points")
    if final.shape[1] != gt.shape[1] or len(gt) < 2:
        raise ValueError("Incompatible GT")
    k = len(final)
    s_k = float(baselines.get_resolution_factor_k(gt, k, seed=0))
    if not np.isfinite(s_k) or s_k <= 0:
        raise ValueError("Undefined FPS resolution for archived GT")

    u_ref = baselines.get_ref_uk(gt, k, seed=0)
    centroids, _ = baselines.get_ref_clusters(gt, c=32, seed=0)
    labels = np.argmin(baselines.cdist(u_ref, centroids), axis=1)
    ref_hist = np.bincount(labels, minlength=len(centroids)).astype(float)
    ref_hist /= ref_hist.sum()

    # Raw coordinates are those in historical CSVs, with no automatic
    # remapping to a distinct registered/normalized Q-score baseline.
    # ideal_res=0 is an *uncorrected* physical measure, not a measured
    # analytical residual: its provenance is explicitly recorded below.
    metrics = {
        "CLOSENESS": float(fair.closeness(final, ref=gt, s_k=s_k,
                                         ideal_res=0.0).value),
        "COVERAGE": float(fair.coverage(final, ref=gt).value),
        "GAP": float(fair.gap(final, ref=gt).value),
        "REGULARITY": float(fair.regularity(final, ref_distribution=u_ref).value),
        "BALANCE": float(fair.balance(final, centroids=centroids,
                                      ref_hist=ref_hist).value),
        "HEADWAY": None,
    }

    if initial is not None:
        initial = np.asarray(initial, dtype=float)
        if initial.ndim != 2 or initial.shape[1] != gt.shape[1]:
            raise ValueError("Mismatched genuine generation-zero population")
        headway = float(fair.headway(final, ref=gt, s_k=s_k,
                                     initial_data=initial).value)
        if np.isfinite(headway):
            metrics["HEADWAY"] = headway

    if not all(np.isfinite(v) for v in metrics.values() if v is not None):
        raise ValueError("Nonfinite FAIR metric")
    return {
        "k": k,
        "s_k": s_k,
        "metrics": metrics,
        "headway_status": ("measured_from_gen0" if metrics["HEADWAY"] is not None
                            else "unavailable_no_generation_zero"),
        "closeness_residue_status": "uncorrected_no_independent_ideal_for_archived_gt",
        "q_scores": None,
        "q_status": "unavailable_no_compatible_calibration",
    }


def compute_archive(data_dir: Path, gt_dir: Path, *,
                    study_filter=None, max_runs=None) -> dict:
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive")
    cache = {}
    runs = []
    errors = []
    for path in sorted(data_dir.glob("*_standard_run*.csv")):
        match = RUN_PATTERN.fullmatch(path.name)
        if match is None or match["gen"] is not None:
            continue
        study = match["study"]
        if study_filter and study != study_filter:
            continue
        # Names convention: DTLZ1_NSGA2, DTLZ1_NSGA3, DTLZ1_MOEAD.
        # A problem name or algorithm containing underscores requires an
        # explicit mapping and is deliberately unsupported here.
        if study.count("_") != 1:
            errors.append({"file": path.name, "reason": "unrecognized study naming"})
            continue
        problem, algorithm = study.split("_", 1)
        gt_file = gt_dir / f"{problem}_3_optimal.csv"
        if not gt_file.is_file():
            errors.append({"file": path.name, "reason": "missing matching 3D GT"})
            continue
        try:
            if problem not in cache:
                cache[problem] = _read_front(gt_file, header=False)
            gt = cache[problem]
            front = _read_front(path, header=True)
            initial_path = data_dir / f"{study}_standard_run{match['run']}_gen0.csv"
            initial = (_read_front(initial_path, header=True)
                       if initial_path.is_file() else None)
            result = compute_fair_only(front, gt, initial)
        except (ValueError, OSError, IndexError) as exc:
            errors.append({"file": path.name, "reason": str(exc)})
            continue
        runs.append({
            "study": study,
            "problem": problem,
            "algorithm": algorithm,
            "run": match["run"],
            "source": path.name,
            "gt_source": gt_file.name,
            "gt_sha256": hashlib.sha256(np.ascontiguousarray(gt).tobytes()).hexdigest(),
            "front_rows": len(front),
            **result,
        })
        if max_runs is not None and len(runs) >= max_runs:
            break

    return {
        "status": "provisional_noncertified",
        "reference_coordinates": "historical_csv_raw",
        "metrics_available": list(METRIC_NAMES),
        "unit": "raw objective coordinates; closeness in FPS resolution units",
        "cannot_be_used_as_certified_audit_report": True,
        "runs": runs,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "calibration/data/calibration_data")
    parser.add_argument("--gt-dir", type=Path,
                        default=ROOT / "calibration/data/ground_truth")
    parser.add_argument("--study", help="Exact study, e.g. DTLZ1_NSGA2")
    parser.add_argument("--max-runs", type=int, default=2,
                        help="Cap successful runs; omit only with --all-runs")
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    forbidden = (ROOT / "calibration/reports").resolve()
    if output.is_relative_to(forbidden) or output == (ROOT / "calibration/audit_report.html").resolve():
        parser.error("Provisional data must not overwrite or appear among certified reports")
    if output.exists():
        parser.error("Refusing to overwrite an existing provisional file")
    result = compute_archive(
        args.data_dir, args.gt_dir, study_filter=args.study,
        max_runs=None if args.all_runs else args.max_runs,
    )
    if not result["runs"]:
        parser.error("No valid runs; review errors and source files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "valid_runs": len(result["runs"]),
        "errors": len(result["errors"]),
        "missing_headway": sum(
            r["metrics"]["HEADWAY"] is None for r in result["runs"]
        ),
        "first_run": result["runs"][0],
    }, indent=2))


if __name__ == "__main__":
    main()
