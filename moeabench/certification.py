# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Versioned, explicit scientific certification and historical baseline tools.

The baseline is never changed by a performance check. Performance decisions
are binary; insufficient power/evidence has a recorded diagnostic reason.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from scipy import stats

SCHEMA = "moeabench-performance-reference"
FAMILY_ALPHA = 0.01
TARGET_POWER = 0.95
MARGIN_FRACTION = 0.001
NUMERIC_FLOOR = 1e-12
PILOT_SEEDS = 8
MAX_SEEDS = 64
FRONT_POINTS = 512
POPULATION = 48
GENERATIONS = 60
MIN_CONFIRMATORY = 8
CASES = (
    ("dtlz2-nsga2", "DTLZ2", {"M": 3}, "NSGA2"),
    ("dtlz2-nsga3", "DTLZ2", {"M": 3}, "NSGA3"),
    ("dtlz2-moead", "DTLZ2", {"M": 3}, "MOEAD"),
    ("dtlz7-nsga2", "DTLZ7", {"M": 3}, "NSGA2"),
    ("dtlz7-nsga3", "DTLZ7", {"M": 3}, "NSGA3"),
    ("dpf1-nsga3", "DPF1", {"M": 3, "D": 2}, "NSGA3"),
)
_SEEDS = tuple(424242 + 104729 * i for i in range(MAX_SEEDS))
_RESOURCE_ROOT = Path(__file__).parent / "resources" / "certifications"
_NUMERICAL_FILES = (
    "tests/stability_reference_fronts.json",
    "tests/stability_reference_gt.json",
    "tests/conventional_metric_reference_v1.json",
)


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_once(destination, document):
    """Atomic-ish immutable creation; never replace an existing certificate."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return path


def _revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _reference_runtime():
    """Fail closed unless the *effective* native reference runtime is tested."""
    from .environment_verification import verify_version, snapshot
    result = verify_version(show=False, force=True)
    if result["status"] != "MATCH":
        raise RuntimeError("Certified numerical environment mismatch: " + str(result["differences"]))
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_numerical_reference_runtime.py"
    if not script.is_file():
        raise RuntimeError("Numerical kernel preflight script is unavailable")
    subprocess.run([sys.executable, str(script)], check=True)
    return snapshot()


def _reference_matrix(problem):
    matrix = np.ascontiguousarray(np.asarray(problem.pf(n_points=FRONT_POINTS), dtype=np.float64))
    if matrix.shape != (FRONT_POINTS, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid fixed ground truth reference points")
    digest = hashlib.sha256(matrix.tobytes()).hexdigest()
    return {"points": matrix.tolist(), "sha256_float64": digest, "shape": [FRONT_POINTS, 3]}


def _validate_fronts(fronts):
    for name in {case[1] for case in CASES}:
        if name not in fronts:
            raise ValueError("Missing frozen reference front: " + name)
        front = fronts[name]
        arr = np.ascontiguousarray(np.asarray(front["points"], dtype=np.float64))
        if arr.shape != (FRONT_POINTS, 3) or not np.isfinite(arr).all():
            raise ValueError("Invalid frozen front: " + name)
        if hashlib.sha256(arr.tobytes()).hexdigest() != front["sha256_float64"]:
            raise ValueError("Frozen reference front checksum mismatch: " + name)


def _measure(case, seed, front):
    import moeabench as mb
    algorithm = getattr(mb.moeas, case["moea"])(**case["moea_kwargs"], seed=int(seed))
    problem = getattr(mb.mops, case["mop"])(**case["mop_kwargs"])
    exp = mb.experiment(mop=problem, moea=algorithm)
    exp.run(repeat=1, silent=True)
    observed = np.asarray(exp.front(), dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 3 or not observed.size:
        raise ValueError("Invalid solver front: " + case["id"])
    if not np.isfinite(observed).all():
        raise ValueError("Nonfinite solver front: " + case["id"])
    result = {"seed": int(seed), "front_size": int(len(observed))}
    for name in ("igdplus", "gdplus", "emd"):
        value = float(getattr(mb.metrics, name)(observed, ref=front, progress=False))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid " + name + " for " + case["id"])
        result[name] = value
    return result


def _default_cases():
    return [
        {
            "id": case_id,
            "mop": mop,
            "mop_kwargs": mop_kwargs,
            "moea": moea,
            "moea_kwargs": {"population": POPULATION, "generations": GENERATIONS},
            "metric": "igdplus",
            "reference_id": mop,
            "seeds": list(_SEEDS),
            "observations": [],
        }
        for case_id, mop, mop_kwargs, moea in CASES
    ]


def create_performance_baseline(output_path):
    """Run an independently approved study; produce an immutable candidate artifact.

    Approval is a separate human review, not inferred from generating a file.
    """
    from .system import version
    if version(show=False) != "0.18.0":
        raise ValueError("The inaugural baseline must use certified MoeaBench 0.18.0")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Historical baseline already exists: " + str(output))
    environment = _reference_runtime()
    import moeabench as mb
    reference_fronts = {}
    for _, mopname, kwargs, _ in CASES:
        if mopname not in reference_fronts:
            reference_fronts[mopname] = _reference_matrix(getattr(mb.mops, mopname)(**kwargs))
    result = {
        "schema": SCHEMA,
        "status": "CANDIDATE_REQUIRES_REVIEW",
        "source_version": version(show=False),
        "source_revision": _revision(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "platform": platform.platform(),
        "numerical_profile": {
            name: os.environ.get(name) for name in (
                "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE",
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
            )
        },
        "protocol": {
            "family_alpha": FAMILY_ALPHA,
            "target_power": TARGET_POWER,
            "margin_fraction": MARGIN_FRACTION,
            "numeric_floor": NUMERIC_FLOOR,
            "pilot": PILOT_SEEDS,
            "max_seeds": MAX_SEEDS,
            "front_points": FRONT_POINTS,
            "decision_metric": "igdplus",
            "diagnostics": ["gdplus", "emd"],
        },
        "reference_fronts": reference_fronts,
        "cases": _default_cases(),
    }
    _validate_fronts(result["reference_fronts"])
    for case in result["cases"]:
        ref = np.asarray(reference_fronts[case["reference_id"]]["points"], dtype=np.float64)
        for seed in case["seeds"]:
            observation = _measure(case, seed, ref)
            case["observations"].append(observation)
            print("BASELINE", case["id"], seed, "igdplus", observation["igdplus"], flush=True)
    return _write_once(output, result)


def load_performance_reference(path, *, require_approved=True):
    reference = json.loads(Path(path).read_text(encoding="utf-8"))
    if reference.get("schema") != SCHEMA:
        raise ValueError("Unsupported certification reference schema")
    if require_approved and reference.get("status") != "APPROVED":
        raise ValueError("Performance reference requires explicit approval")
    if reference.get("source_version") != "0.18.0":
        raise ValueError("Unexpected historical reference version")
    if not isinstance(reference.get("source_revision"), str) or not reference["source_revision"]:
        raise ValueError("Missing historical source revision")
    _validate_fronts(reference["reference_fronts"])
    expected = _default_cases()
    got = reference.get("cases")
    if not isinstance(got, list) or len(got) != len(expected):
        raise ValueError("Performance reference must contain all six planned cases")
    for actual, planned in zip(got, expected):
        for key in ("id", "mop", "moea", "mop_kwargs", "moea_kwargs", "seeds", "reference_id"):
            if actual.get(key) != planned[key]:
                raise ValueError("Unapproved performance configuration change: " + key)
        obs = actual.get("observations")
        if not isinstance(obs, list) or len(obs) != MAX_SEEDS:
            raise ValueError("Reference must contain 64 observations per case")
        for seed, item in zip(_SEEDS, obs):
            if item.get("seed") != seed:
                raise ValueError("Performance reference seed mismatch")
            for metric in ("igdplus", "gdplus", "emd"):
                if not math.isfinite(float(item[metric])) or float(item[metric]) < 0:
                    raise ValueError("Nonfinite or negative reference measurement")
    return reference


def _required_confirmatory(pilot_sd, margin, num_cases):
    """Noncentral Student-t design; power at mean difference == 0."""
    if pilot_sd < 0 or not math.isfinite(pilot_sd):
        raise ValueError("Invalid pilot standard deviation")
    if pilot_sd <= 32 * np.finfo(float).eps:
        return MIN_CONFIRMATORY
    tail_alpha = FAMILY_ALPHA / (2 * num_cases)
    for n in range(MIN_CONFIRMATORY, MAX_SEEDS - PILOT_SEEDS + 1):
        tcrit = stats.t.ppf(1 - tail_alpha, n - 1)
        shift = -margin * math.sqrt(n) / pilot_sd
        power = stats.nct.cdf(-tcrit, n - 1, shift)
        if power >= TARGET_POWER:
            return n
    return None


def assess_paired(case, pilot_scores, confirmatory_scores, baseline):
    """Pure comparison function with pilot excluded from the final interval."""
    values = np.asarray(baseline, dtype=float)
    pilot = np.asarray(pilot_scores, dtype=float)
    if len(values) != MAX_SEEDS or len(pilot) != PILOT_SEEDS:
        raise ValueError("Expected 64 reference and eight pilot observations")
    if not np.isfinite(values).all() or not np.isfinite(pilot).all():
        raise ValueError("Nonfinite observations")
    margin = max(NUMERIC_FLOOR, MARGIN_FRACTION * float(np.mean(values)))
    sd = float(np.std(pilot - values[:PILOT_SEEDS], ddof=1))
    n = _required_confirmatory(sd, margin, len(CASES))
    if n is None:
        return {"id": case["id"], "decision": "NOT_CERTIFIED",
                "reason": "INSUFFICIENT_POWER", "pilot_sd": sd,
                "margin": margin, "required_confirmatory": None}
    observed = np.asarray(confirmatory_scores, dtype=float)
    if len(observed) != n or not np.isfinite(observed).all():
        raise ValueError("Confirmatory observations do not match required sample")
    diff = observed - values[PILOT_SEEDS:PILOT_SEEDS + n]
    mean = float(np.mean(diff))
    stderr = float(np.std(diff, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1 - FAMILY_ALPHA / (2 * len(CASES)), n - 1))
    lower, upper = mean - tcrit * stderr, mean + tcrit * stderr
    good = upper < margin
    reason = ("NONINFERIOR" if good else
              "REGRESSION_DETECTED" if lower > margin else "INSUFFICIENT_EVIDENCE")
    return {"id": case["id"], "decision": "CERTIFIED" if good else "NOT_CERTIFIED",
            "reason": reason, "pilot_sd": sd, "margin": margin,
            "confirmatory_runs": n, "mean_difference": mean,
            "ci_lower": lower, "ci_upper": upper,
            "pilot_runs": PILOT_SEEDS, "power_target": TARGET_POWER,
            "family_alpha": FAMILY_ALPHA}


def certify_performance(reference_path, output_path):
    """Compare with a frozen approved historical reference; never change it."""
    reference = load_performance_reference(reference_path)
    from .system import version
    if version(show=False) == reference["source_version"]:
        raise ValueError("Performance certification requires a candidate version distinct from the historical baseline")
    current_environment = _reference_runtime()
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Performance certificate already exists")
    reports = []
    for case in reference["cases"]:
        front = np.asarray(reference["reference_fronts"][case["reference_id"]]["points"], dtype=float)
        baseline = [o["igdplus"] for o in case["observations"]]
        pilot = [_measure(case, seed, front)["igdplus"] for seed in case["seeds"][:PILOT_SEEDS]]
        margin = max(NUMERIC_FLOOR, MARGIN_FRACTION * float(np.mean(baseline)))
        sd = float(np.std(np.asarray(pilot) - np.asarray(baseline[:PILOT_SEEDS]), ddof=1))
        required = _required_confirmatory(sd, margin, len(CASES))
        confirmatory = []
        if required is not None:
            confirmatory = [
                _measure(case, seed, front)["igdplus"]
                for seed in case["seeds"][PILOT_SEEDS:PILOT_SEEDS + required]
            ]
        report = assess_paired(case, pilot, confirmatory, baseline)
        report["pilot_scores"] = pilot
        report["confirmatory_scores"] = confirmatory
        reports.append(report)
        print("CERTIFICATION", case["id"], report["decision"], report["reason"], flush=True)
    status = ("CERTIFIED" if all(r["decision"] == "CERTIFIED" for r in reports)
              else "NOT_CERTIFIED")
    result = {
        "schema": "moeabench-performance-certificate",
        "status": status,
        "baseline_sha256": _hash_file(reference_path),
        "baseline_source_revision": reference["source_revision"],
        "candidate_source_revision": _revision(),
        "candidate_environment": current_environment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": reports,
    }
    return _write_once(output, result), result


def register_numeric_history(root=None, *, evidence_url=None, counts=None):
    """Record a compact fingerprint; do not copy numerical front arrays."""
    from .environment_verification import snapshot, certified_manifest
    from .system import version
    manifest = certified_manifest()
    if manifest["versions"] != snapshot():
        raise RuntimeError("Current environment does not match current certified manifest")
    repo_root = Path(__file__).resolve().parents[1]
    digests = {}
    for relative in _NUMERICAL_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise RuntimeError("Missing numeric regression reference: " + relative)
        digests[relative] = _hash_file(path)
    target = Path(root) if root else _RESOURCE_ROOT
    output = target / version(show=False) / "numeric-certificate.json"
    document = {
        "schema": "moeabench-numeric-certificate",
        "version": version(show=False),
        "source_revision": _revision(),
        "environment": snapshot(),
        "platform": platform.platform(),
        "numerical_profile": {
            name: os.environ.get(name) for name in (
                "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE",
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
            )
        },
        "numerical_fixtures_sha256": digests,
        "evidence_url": evidence_url,
        "counts": counts,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return _write_once(output, document)


def certification_history(root=None):
    """Enumerate recorded certificates; never infer numerical agreement from versions."""
    base = Path(root) if root is not None else _RESOURCE_ROOT
    results = []
    if not base.exists():
        return results
    for path in sorted(base.glob("*/numeric-certificate.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != "moeabench-numeric-certificate":
            raise ValueError("Invalid stored numerical certificate: " + str(path))
        performance_records = []
        for performance_path in sorted(path.parent.glob("performance-vs-*.json")):
            performance = json.loads(performance_path.read_text(encoding="utf-8"))
            if performance.get("schema") != "moeabench-performance-certificate":
                raise ValueError("Invalid stored performance certificate: " + str(performance_path))
            performance_records.append({
                "status": performance["status"],
                "baseline_source_revision": performance["baseline_source_revision"],
                "candidate_source_revision": performance["candidate_source_revision"],
                "file": performance_path.name,
            })
        results.append({
            "version": record["version"], "source_revision": record["source_revision"],
            "environment": record["environment"],
            "numerical_fixtures_sha256": record["numerical_fixtures_sha256"],
            "evidence_url": record.get("evidence_url"), "numeric_certified": True,
            "performance_certifications": performance_records,
        })
    return results


def matching_certified_environments(current=None, root=None):
    """Report exact version fingerprint matches, not numerical equivalence claims."""
    from .environment_verification import snapshot
    fingerprint = snapshot() if current is None else current
    return [entry for entry in certification_history(root)
            if entry["environment"] == fingerprint]
