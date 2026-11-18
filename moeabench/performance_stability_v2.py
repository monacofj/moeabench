# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #98: conservative, preregistered paired-performance diagnostic protocol.

A three-way inference about the *mean* paired change relative to an explicit
practical degradation margin. The opt-in v2 schema is not a release gate.
Existing v1 users and regression expectations remain unchanged.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


SCHEMA_V2 = "moeabench_performance_reference_v2"
MIN_RUNS = 8
MAX_POWER_PLANNING_RUNS = 5000
METRICS = frozenset(("igd", "igdplus", "gd", "gdplus", "emd"))


def _fraction(value: Any, name: str, *, lower: float = 0.0, upper: float = 1.0) -> float:
    number = float(value)
    if not np.isfinite(number) or not lower < number < upper:
        raise ValueError(f"{name} must be finite and strictly between {lower} and {upper}")
    return number


def _nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def practical_margin(case: dict) -> float:
    baseline = np.asarray(case["baseline_values"], dtype=float)
    floor = max(
        _nonnegative(case["max_absolute_degradation"], "max_absolute_degradation"),
        _nonnegative(case["max_relative_degradation"], "max_relative_degradation")
        * float(np.mean(baseline)),
    )
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError(f"{case['id']}: v2 requires a strictly positive practical margin")
    return floor


def minimum_required_runs(margin: float, pilot_sd: float, family_alpha: float,
                          target_power: float, family_cases: int) -> int:
    """Student t power under paired normal differences, with preregistered pilot SD.

    Each of the two confidence-bound decisions per case receives
    alpha_family/(2 * number_of_cases). Under the model with true paired
    improvement 0 and noninferiority boundary +margin, the noncentral t CDF
    yields power to conclude noninferiority. By symmetry this is also the
    power to detect a true regression of +2*margin.
    """
    if family_cases < 1:
        raise ValueError("A performance family must have at least one case")
    margin = _positive(margin, "margin")
    pilot_sd = _nonnegative(pilot_sd, "pilot_sd")
    family_alpha = _fraction(family_alpha, "family_alpha")
    target_power = _fraction(target_power, "target_power")
    if pilot_sd == 0:
        return MIN_RUNS
    tail_alpha = family_alpha / (2 * family_cases)
    for n in range(MIN_RUNS, MAX_POWER_PLANNING_RUNS + 1):
        tcrit = float(stats.t.ppf(1 - tail_alpha, df=n - 1))
        shift = -margin * math.sqrt(n) / pilot_sd
        power = float(stats.nct.cdf(-tcrit, df=n - 1, nc=shift))
        if np.isfinite(power) and power >= target_power:
            return n
    raise ValueError(
        f"Pilot uncertainty implies more than {MAX_POWER_PLANNING_RUNS} paired runs; "
        "no automatic exhaustive experiment is authorized"
    )


def validate_v2(doc: dict) -> dict:
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_V2:
        raise ValueError("Unsupported v2 statistical performance reference")
    provenance = doc.get("provenance")
    required = (
        "source_revision", "reference_origin", "reviewed_by",
        "environment_fingerprint", "measurement_protocol", "pilot_origin",
    )
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(name), str) or not provenance[name].strip()
        for name in required
    ):
        raise ValueError("v2 requires independently reviewable provenance and pilot metadata")
    cases = doc.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("v2 requires at least one explicit performance case")
    family_alpha = _fraction(doc.get("family_alpha"), "family_alpha")
    target_power = _fraction(doc.get("target_power"), "target_power")
    if not 0.8 <= target_power <= 0.99:
        raise ValueError("target_power must be preregistered between 0.80 and 0.99")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip():
            raise ValueError("Each v2 case needs a nonempty id")
        name = case["id"]
        if name in seen:
            raise ValueError(f"Duplicate performance case: {name}")
        seen.add(name)
        for element in ("mop", "moea"):
            obj = case.get(element)
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
                raise ValueError(f"{name}: {element}.name is required")
            if not isinstance(obj.get("kwargs", {}), dict):
                raise ValueError(f"{name}: {element}.kwargs must be a dictionary")
        if "seed" in case["moea"].get("kwargs", {}):
            raise ValueError(f"{name}: seed must be supplied only in seeds")
        if case.get("metric") not in METRICS:
            raise ValueError(f"{name}: unsupported minimized quality metric")
        ref = np.asarray(case.get("reference_front"), dtype=float)
        if ref.ndim != 2 or not ref.size or not np.isfinite(ref).all():
            raise ValueError(f"{name}: reference_front must be a finite 2D matrix")
        specified_dimension = case["mop"].get("kwargs", {}).get("M")
        if specified_dimension is not None and specified_dimension != ref.shape[1]:
            raise ValueError(f"{name}: objective dimension mismatch")
        seeds = case.get("seeds")
        values = case.get("baseline_values")
        if not isinstance(seeds, list) or not all(
            isinstance(s, int) and not isinstance(s, bool) and 0 <= s < 2**32
            for s in seeds
        ) or len(set(seeds)) != len(seeds):
            raise ValueError(f"{name}: seeds must be unique unsigned 32-bit integers")
        if not isinstance(values, list) or len(values) != len(seeds):
            raise ValueError(f"{name}: one baseline observation per seed is required")
        baseline = np.asarray(values, dtype=float)
        if not np.isfinite(baseline).all() or (baseline < 0).any():
            raise ValueError(f"{name}: baseline values must be finite and nonnegative")
        if "alpha" in case:
            raise ValueError(f"{name}: v2 uses only family_alpha, not per-case alpha")
        rationale = case.get("sample_size_rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{name}: a preregistered sample_size_rationale is required")
        pilot_sd = _nonnegative(case.get("pilot_standard_deviation"), "pilot_standard_deviation")
        needed = minimum_required_runs(
            practical_margin(case), pilot_sd, family_alpha, target_power, len(cases)
        )
        if len(seeds) < needed:
            raise ValueError(
                f"{name}: {len(seeds)} paired seeds are insufficient for the "
                f"preregistered pilot power (minimum {needed})"
            )
    return doc


def assess_case_v2(case: dict, observed: list[float], *,
                   family_alpha: float, family_cases: int,
                   target_power: float) -> dict:
    baseline = np.asarray(case["baseline_values"], dtype=float)
    current = np.asarray(observed, dtype=float)
    if (current.shape != baseline.shape or not np.isfinite(current).all()
            or (current < 0).any()):
        raise ValueError("Current measurements must be finite nonnegative paired observations")
    margin = practical_margin(case)
    required = minimum_required_runs(
        margin, _nonnegative(case["pilot_standard_deviation"], "pilot_standard_deviation"),
        family_alpha, target_power, family_cases,
    )
    if len(current) < required:
        raise ValueError(f"{case['id']}: insufficient paired runs for the preregistered power")

    differences = current - baseline
    if not np.isfinite(differences).all():
        raise ValueError(f"{case['id']}: nonfinite paired differences")
    n = len(current)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    stderr = sd / math.sqrt(n)
    tail_alpha = family_alpha / (2 * family_cases)
    tcrit = float(stats.t.ppf(1 - tail_alpha, df=n - 1))
    ci_low, ci_high = mean - tcrit * stderr, mean + tcrit * stderr
    if ci_high < margin:
        status = "PASSED"
    elif ci_low > margin:
        status = "FAILED"
    else:
        status = "INCONCLUSIVE"
    return {
        "id": case["id"],
        "status": status,
        "runs": n,
        "required_runs": required,
        "baseline_mean": float(np.mean(baseline)),
        "observed_mean": float(np.mean(current)),
        "mean_delta": mean,
        "practical_floor": margin,
        "paired_sd": sd,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "family_alpha": family_alpha,
        "per_tail_alpha": tail_alpha,
        "target_power": target_power,
    }


def aggregate_v2(reports: list[dict]) -> str:
    if any(r["status"] == "FAILED" for r in reports):
        return "FAILED"
    if any(r["status"] == "INCONCLUSIVE" for r in reports):
        return "INCONCLUSIVE"
    return "PASSED"
