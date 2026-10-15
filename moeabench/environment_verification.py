# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime environment provenance and explicitly requested numerical verification.

Version matching is evidence of compatibility, not proof of numerical stability.
Verification state is scoped to the current interpreter and environment fingerprint;
the canonical reference is shipped with the package, not regenerated at runtime.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import warnings

DEPENDENCIES = ("numpy", "scipy", "pymoo", "pandas", "moocore", "deap")
_MANIFEST_PATH = Path(__file__).parent / "diagnostics" / "resources" / "certified_environment.json"
_FRONTS_PATH = Path(__file__).parent / "diagnostics" / "resources" / "stability_smoke.json"
_version_cache = None
_stability_cache = None
_warned_fingerprints = set()


class EnvironmentWarning(UserWarning):
    """Environment provenance is unverified or differs from the certified reference."""


def snapshot():
    """Capture a version fingerprint (no timestamps: suitable for comparison)."""
    from .system import version
    versions = {"moeabench": version(show=False), "python": platform.python_version()}
    for dependency in DEPENDENCIES:
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = None
    return versions


def certified_manifest():
    with _MANIFEST_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema") != 1:
        raise ValueError("Unsupported certified environment manifest schema")
    return payload


def verify_version(show=False, force=False):
    """Compare with the developer-certified versions, memoized for this environment.

    A mismatch is not a numerical failure. A manifest with no certified versions
    is reported as UNCONFIGURED; no ambient installation is silently trusted.
    """
    global _version_cache
    current = snapshot()
    manifest = certified_manifest()
    fingerprint = (tuple(sorted(current.items())), json.dumps(manifest, sort_keys=True))
    if not force and _version_cache and _version_cache["fingerprint"] == fingerprint:
        result = _version_cache["result"]
    else:
        reference = manifest.get("versions", {})
        mismatches = {name: {"certified": value, "current": current.get(name)}
                      for name, value in reference.items() if current.get(name) != value}
        status = "UNCONFIGURED" if not reference else ("MISMATCH" if mismatches else "MATCH")
        result = {"status": status, "differences": mismatches, "current": current,
                  "certified": dict(reference)}
        _version_cache = {"fingerprint": fingerprint, "result": result}
    if show:
        print("MoeaBench certified version check:", result["status"])
        for name, values in result["differences"].items():
            print("  %s: certified=%s current=%s" % (name, values["certified"], values["current"]))
    return result


def version_status():
    """Retrieve cached result only; reports must not initiate verification."""
    return _version_cache["result"]["status"] if _version_cache else "UNCHECKED"


def stability_status():
    current = snapshot()
    if _stability_cache and _stability_cache["fingerprint"] == current:
        return _stability_cache["result"]["status"]
    return "UNCHECKED"


def verify_experiment(exp, show=False):
    """Compare the current environment with provenance captured at execution time."""
    original = getattr(exp, "_execution_environment", None)
    current = snapshot()
    if not original:
        result = {"status": "UNKNOWN", "differences": {}}
    else:
        differences = {key: {"original": val, "current": current.get(key)}
                       for key, val in original.items() if current.get(key) != val}
        result = {"status": "MISMATCH" if differences else "MATCH", "differences": differences}
    # This is descriptive and never changes the global stability certification.
    exp._environment_comparison = result
    if show:
        print("Experiment environment:", result["status"])
    return result


def verify_stability(show=True, atol=1e-12, rtol=1e-8):
    """Check exact PRNG signatures and bounded numerical reference fronts.

    This is manually invoked and never modifies the certified manifest.
    Unlike development pytest tests, it works in an installed distribution.
    """
    global _stability_cache
    import numpy as np
    from . import mops, moeas
    from .core.experiment import experiment
    results = {}

    def check(name, condition):
        results[name] = "PASS" if condition else "FAIL"

    python_rng = random.Random(7)
    check("python_random", [python_rng.random() for _ in range(5)] ==
          [0.32383276483316237, 0.15084917392450192,
           0.6509344730398537, 0.07243628666754276,
           0.5358820043066892])
    check("numpy_randomstate",
          np.array_equal(np.random.RandomState(7).random_sample(5),
                         np.array([0.07630828937395717, 0.7799187922401146,
                                   0.4384092314408935, 0.7234651778309412,
                                   0.9779895119966027])))
    check("numpy_randomstate_ints",
          np.array_equal(np.random.RandomState(7).randint(0, 1000, size=5),
                         np.array([175, 196, 537, 502, 579])))
    check("numpy_generator",
          np.array_equal(np.random.default_rng(7).random(5),
                         np.array([0.625095466604667, 0.8972138009695755,
                                   0.7756856902451935, 0.22520718999059186,
                                   0.30016628491122543])))
    check("numpy_generator_normal",
          np.array_equal(np.random.default_rng(7).normal(size=5),
                         np.array([0.0012301533574825742, 0.2987455375084699,
                                   -0.2741378553622176, -0.8905918387572742,
                                   -0.45467078517172255])))
    check("numpy_generator_ints",
          np.array_equal(np.random.default_rng(7).integers(0, 1000, size=5),
                         np.array([944, 625, 684, 897, 578])))
    with _FRONTS_PATH.open(encoding="utf-8") as f:
        reference = json.load(f)
    if reference.get("schema") != 1:
        raise ValueError("Unsupported numerical stability reference schema")

    for entry in reference["smoke"]:
        label = entry["algorithm"] + "/" + entry["mop"]
        try:
            mop_class = getattr(mops, entry["mop"])
            try:
                problem = mop_class(M=3)
            except TypeError:
                problem = mop_class(M=3, D=2)
            algorithm = getattr(moeas, entry["algorithm"])(
                seed=entry["seed"], population=entry["population"],
                generations=entry["generations"])
            exp = experiment(mop=problem, moea=algorithm)
            exp.run(repeat=1, silent=True)
            observed = np.asarray(exp.front())
            expected = np.asarray(entry["front"], dtype=float)
            if observed.shape == expected.shape:
                observed = observed[np.lexsort(observed.T[::-1])]
                expected = expected[np.lexsort(expected.T[::-1])]
                matched = bool(np.allclose(observed, expected, atol=atol, rtol=rtol))
            else:
                matched = False
            check(label, matched)
        except Exception as exc:
            results[label] = "ERROR: " + str(exc)

    status = "PASSED" if all(v == "PASS" for v in results.values()) else "FAILED"
    result = {"status": status, "checks": results, "environment": snapshot()}
    _stability_cache = {"fingerprint": snapshot(), "result": result}
    if show:
        print("MoeaBench numerical stability:", status)
        for key, value in results.items():
            print("  %-28s %s" % (key, value))
    return result


def environment_report(exp=None):
    """Non-mutating summary suitable for experiment reports."""
    system_state = version_status()
    stability_state = stability_status()
    experiment_state = (getattr(exp, "_environment_comparison", None) or {}).get(
        "status", "UNCHECKED") if exp is not None else "N/A"
    return {"versions": system_state, "stability": stability_state,
            "experiment": experiment_state}


def warn_on_unverified():
    """A single advisory at first run, not a warning on every report."""
    # exp.run() uses the cached version check after the first invocation.
    # Explicit verify_version(force=True) is available if the environment changes.
    result = (_version_cache or {}).get("result") or verify_version(show=False)
    fingerprint = tuple(sorted(result["current"].items()))
    if result["status"] != "MATCH" and stability_status() != "PASSED" and fingerprint not in _warned_fingerprints:
        _warned_fingerprints.add(fingerprint)
        warnings.warn("MoeaBench environment %s relative to certified versions; "
                      "run mb.system.verify_stability() to check numerical behavior."
                      % result["status"], EnvironmentWarning, stacklevel=3)
    return result
