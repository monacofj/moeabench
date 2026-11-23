# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact continuous-front HV denominators for original DTLZ1 and DTLZ2.

Independent mathematical research for issue #89; NOT a public HV API.
No calibration sidecars, existing tests or certificates are changed.

These formulas require *minimization*, the original complete continuous
Pareto fronts, the shared exact axis normalization, and r>=1 in every
normalized objective. They do NOT apply automatically to other MOPs or
to externally supplied/non-axis normalization contexts.

Run: python -m scripts.audit.hv_analytic_dtlz_issue89
"""

from __future__ import annotations

import json
from math import factorial, gamma, pi
import numpy as np


def continuous_hv(problem: str, objectives: int, reference: float = 1.1) -> float:
    """Hypervolume of the exact continuous normalized DTLZ1/2 PF.

    The nondominated complement in the reference cube [0,r]^M is:
    - DTLZ1: simplex {u_i>=0, sum u_i < 1}, volume 1/M!;
    - DTLZ2: positive unit M-ball {u_i>=0, sum u_i**2 < 1},
      volume pi**(M/2) / (2**M Gamma(M/2+1)).
    For r>=1 these bodies are fully contained in the reference cube.
    """
    if not isinstance(objectives, int) or objectives < 2:
        raise ValueError("Objectives must be an integer >=2")
    if not np.isfinite(reference) or reference < 1:
        raise ValueError("The complete unit complement requires r >= 1")
    if problem == "DTLZ1":
        nondominated = 1 / factorial(objectives)
    elif problem == "DTLZ2":
        nondominated = pi ** (objectives / 2) / (
            2 ** objectives * gamma(objectives / 2 + 1)
        )
    else:
        raise ValueError("Only DTLZ1 and DTLZ2 exact PFs are proven")
    return float(reference ** objectives - nondominated)


def sample_check(problem: str, objectives: int, points: int = 64) -> dict:
    """Confirm actual GT samples respect the proved exact HV ceiling.

    Uses the public *raw* exact metric with the sample's axis-anchored
    normalization geometry. This does not replace the GT-based H_abs.
    """
    import moeabench as mb

    mop = getattr(mb.mops, problem)(M=objectives)
    gt = np.asarray(mop.pf(n_points=points), dtype=float)
    if gt.shape != (points, objectives):
        raise AssertionError("MOP sampler returned unexpected objective shape")
    if problem == "DTLZ1":
        np.testing.assert_allclose(gt.sum(axis=1), 0.5, atol=1e-9, rtol=0)
        expected_nadir = 0.5
    else:
        np.testing.assert_allclose(np.sum(gt ** 2, axis=1), 1, atol=1e-9, rtol=0)
        expected_nadir = 1.0
    np.testing.assert_allclose(gt.min(axis=0), 0.0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        gt.max(axis=0), expected_nadir, atol=1e-10, rtol=0
    )

    finite = float(mb.metrics.hv(
        gt, ref=gt, mode="exact", scale="raw", progress=False
    ))
    exact = continuous_hv(problem, objectives)
    if not 0 < finite < exact:
        raise AssertionError(
            f"GT sampling bound failed: {problem} M={objectives}: {finite} vs {exact}"
        )
    return {
        "problem": problem,
        "objectives": objectives,
        "points": points,
        "finite_gt_hv": finite,
        "true_continuous_hv": exact,
        "finite_to_exact_ratio": finite / exact,
    }


def main():
    print(json.dumps({
        "study": "issue-89-exact-continuous-hv-dtlz-many-objectives",
        "status": "exploratory; not a public metric or release certificate",
        "analytical_denominators": [
            {"problem": problem, "objectives": m,
             "exact_hv": continuous_hv(problem, m)}
            for problem in ("DTLZ1", "DTLZ2") for m in (2, 3, 4, 5, 8)
        ],
        "actual_sampler_checks": [
            sample_check(problem, m)
            for problem in ("DTLZ1", "DTLZ2") for m in (2, 3, 4)
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
