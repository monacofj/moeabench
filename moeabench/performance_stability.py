# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in, baseline-backed statistical performance regression checks.

This module never generates or updates its own reference measurements.
The user must provide an independently reviewed, versioned reference JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

SCHEMA = "moeabench_performance_reference_v1"
METRICS = frozenset({"igd", "igdplus", "gd", "gdplus", "emd"})


def _nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def load_reference(path: str | Path) -> dict:
    """Validate a *preexisting* reference before executing any algorithm."""
    with Path(path).open(encoding="utf-8") as fh:
        document = json.load(fh)
    if document.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported performance reference schema; expected {SCHEMA}")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict) or not all(
        isinstance(provenance.get(k), str) and provenance[k].strip()
        for k in ("source_revision", "reference_origin", "reviewed_by")
    ):
        raise ValueError("Independent reference provenance and reviewer are required")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("A performance reference must contain at least one case")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("Each case needs a nonempty id")
        if case["id"] in seen:
            raise ValueError(f"Duplicate performance case: {case['id']}")
        seen.add(case["id"])
        for key in ("mop", "moea"):
            item = case.get(key)
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
                raise ValueError(f"{case['id']}: {key}.name is required")
            if not isinstance(item.get("kwargs", {}), dict):
                raise ValueError(f"{case['id']}: {key}.kwargs must be an object")
        if "seed" in case["moea"].get("kwargs", {}):
            raise ValueError(f"{case['id']}: seeds must be specified only in seeds")
        if case.get("metric") not in METRICS:
            raise ValueError(f"{case['id']}: unsupported metric")
        front = np.asarray(case.get("reference_front"), dtype=float)
        if front.ndim != 2 or not front.size or not np.isfinite(front).all():
            raise ValueError(f"{case['id']}: reference_front must be a finite 2D matrix")
        seeds = case.get("seeds")
        values = case.get("baseline_values")
        if not isinstance(seeds, list) or len(seeds) < 3 or not all(
            isinstance(s, int) and not isinstance(s, bool) and 0 <= s < 2**32 for s in seeds
        ) or len(set(seeds)) != len(seeds):
            raise ValueError(f"{case['id']}: at least three distinct integer seeds are required")
        if not isinstance(values, list) or len(values) != len(seeds):
            raise ValueError(f"{case['id']}: one baseline measurement per seed is required")
        baseline = np.asarray(values, dtype=float)
        if not np.isfinite(baseline).all() or (baseline < 0).any():
            raise ValueError(f"{case['id']}: baseline must contain finite nonnegative values")
        alpha = float(case.get("alpha", 0.05))
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError(f"{case['id']}: alpha must be between zero and one")
        for key in ("max_relative_degradation", "max_absolute_degradation"):
            if key not in case:
                raise ValueError(f"{case['id']}: explicit {key} is required")
            _nonnegative(case[key], key)
    return document


def _measure_case(case: dict, seed: int) -> float:
    """Measure a single fixed-seed solver run against the external front."""
    import moeabench as mb

    problem_spec = case["mop"]
    algorithm_spec = case["moea"]
    problem = getattr(mb.mops, problem_spec["name"])(**problem_spec.get("kwargs", {}))
    algorithm = getattr(mb.moeas, algorithm_spec["name"])(
        **algorithm_spec.get("kwargs", {}), seed=seed
    )
    exp = mb.experiment(mop=problem, moea=algorithm)
    exp.run(repeat=1, silent=True)
    observed = np.asarray(exp.front(), dtype=float)
    if observed.ndim != 2 or not observed.size or not np.isfinite(observed).all():
        raise ValueError(f"{case['id']}: solver produced an invalid front")
    ref = np.asarray(case["reference_front"], dtype=float)
    score = float(getattr(mb.metrics, case["metric"])(
        observed, ref=ref, progress=False
    ))
    if not np.isfinite(score) or score < 0:
        raise ValueError(f"{case['id']}: metric must be finite and nonnegative")
    return score


def assess_case(case: dict, observed: list[float]) -> dict:
    """One-sided paired t-test plus an explicit practical-degradation floor.

    All supported metrics are minimized. Seeds pair the new and old runs.
    This test detects increases in mean cost, not arbitrary distribution drift.
    """
    baseline = np.asarray(case["baseline_values"], dtype=float)
    current = np.asarray(observed, dtype=float)
    if (current.shape != baseline.shape or not np.isfinite(current).all()
            or (current < 0).any()):
        raise ValueError("Current measurements must be finite, nonnegative and match the paired reference")
    changes = current - baseline
    mean_delta = float(np.mean(changes))
    practical_floor = max(
        float(case["max_absolute_degradation"]),
        float(case["max_relative_degradation"]) * abs(float(np.mean(baseline))),
    )
    # The zero-variance case is a valid deterministic paired comparison.
    precision_floor = 32 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(changes))))
    if float(np.ptp(changes)) <= precision_floor:
        pvalue = 0.0 if mean_delta > 0 else 1.0
    else:
        pvalue = float(stats.ttest_1samp(changes, popmean=0.0, alternative="greater").pvalue)
    if not np.isfinite(pvalue):
        raise ValueError("Statistical comparison produced a nonfinite p-value")
    failed = mean_delta > practical_floor and pvalue < float(case.get("alpha", 0.05))
    return {
        "id": case["id"],
        "status": "FAILED" if failed else "PASSED",
        "runs": len(current),
        "baseline_mean": float(np.mean(baseline)),
        "observed_mean": float(np.mean(current)),
        "mean_delta": mean_delta,
        "practical_floor": practical_floor,
        "pvalue": pvalue,
    }


def verify_performance(reference_path: str | Path, show: bool = True) -> dict:
    """Run only the explicitly listed scenarios; never create baselines."""
    reference = load_reference(reference_path)
    reports = []
    for case in reference["cases"]:
        observed = [_measure_case(case, seed) for seed in case["seeds"]]
        reports.append(assess_case(case, observed))
    status = "FAILED" if any(r["status"] == "FAILED" for r in reports) else "PASSED"
    result = {"status": status, "cases": reports, "source": str(reference_path)}
    if show:
        print("MoeaBench statistical performance stability:", status)
        for report in reports:
            print("  %s: %s (n=%d, p=%.4g, mean_delta=%.6g)" % (
                report["id"], report["status"], report["runs"],
                report["pvalue"], report["mean_delta"]
            ))
    return result
