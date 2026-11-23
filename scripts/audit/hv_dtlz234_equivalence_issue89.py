# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Research-only check: DTLZ3 and DTLZ4 share DTLZ2's exact full PF.

The equivalence concerns the continuous true Pareto front, *not*
finite-sampler equality or an authorized public H_abs denominator change.
"""
from __future__ import annotations

import json

import numpy as np
import moeabench as mb
from scripts.audit.hv_analytic_dtlz_issue89 import continuous_hv


def exact_dtlz234_hv(objectives: int, reference: float = 1.1) -> float:
    """True continuous normalized HV for original DTLZ2, 3 and 4."""
    return continuous_hv("DTLZ2", objectives, reference=reference)


def observed_front_check(name: str, objectives: int, n: int = 64):
    """Use the actual MOP PF sampler and the exact current public raw engine."""
    if name not in ("DTLZ2", "DTLZ3", "DTLZ4"):
        raise ValueError("Only proven spherical-front problems are supported.")
    mop = getattr(mb.mops, name)(M=objectives)
    sampled_decisions = np.asarray(mop.ps(n_points=n), dtype=float)
    sampled = np.asarray(mop.evaluation(sampled_decisions)["F"], dtype=float)
    if sampled.shape != (n, objectives):
        raise AssertionError("MOP sampling returned unexpected dimensions")
    np.testing.assert_allclose(
        np.sum(sampled**2, axis=1), np.ones(n), rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(sampled.min(axis=0), 0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(sampled.max(axis=0), 1, rtol=0, atol=1e-9)
    raw = float(mb.metrics.hv(
        sampled, ref=sampled, mode="exact", scale="raw", progress=False
    ))
    theoretical = exact_dtlz234_hv(objectives)
    if not (0 < raw < theoretical):
        raise AssertionError("Finite actual front must fall below true HV")
    return {
        "problem": name,
        "M": objectives,
        "points": n,
        "finite_gt_raw_hv": raw,
        "true_continuous_hv": theoretical,
        "ratio_sample_to_true": raw / theoretical,
    }


def main():
    print(json.dumps({
        "study": "issue-89-dtlz234-continuous-hv-equivalence",
        "scope": "analytical front geometry; no production metric changes",
        "cases": [
            observed_front_check(name, m)
            for name in ("DTLZ2", "DTLZ3", "DTLZ4")
            for m in (2, 3, 4)
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
