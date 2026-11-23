# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-geometry deterministic GT refinement research for issue #89.

Research only: no modification of calibration, public metrics, existing
regressions, performance baselines or certification manifests.

Run: python -m scripts.audit.hv_gt_refinement_issue89
"""
from __future__ import annotations

import json

import numpy as np

import moeabench as mb
from scripts.audit.hv_reference_density_issue89 import independent_hv2
from scripts.audit.hv_dpf1_degenerate_issue89 import independent_dpf1_hv3


def independent_hv(name, front, ideal, nadir):
    if name == "DTLZ7":
        return independent_hv2(front, ideal, nadir)
    if name == "DPF1":
        return independent_dpf1_hv3(front, ideal, nadir)
    raise ValueError("Only two independently verified geometric oracles")


def analyze_case(name: str, *, dense_count=2048,
                 budgets=(64, 128, 256, 512, 1024),
                 check_public=False) -> dict:
    """Compare real sampler re-generation with nested reference refinement.

    Both strategies use the *same* pre-fixed objective-space ideal/nadir
    derived from an independent, dense anchor front. Thus apparent
    convergence is not caused by moving the normalization box.

    Independent: direct fresh mop.pf(n) for each n.
    Nested: prefixes of one dense mop.pf(dense_count).
    This is a numerical convergence *diagnostic*, not a proof that either
    series approaches the exact continuous-front volume.
    """
    if name == "DTLZ7":
        mop = mb.mops.DTLZ7(M=2)
    elif name == "DPF1":
        mop = mb.mops.DPF1(M=3, D=2)
    else:
        raise ValueError("Unsupported experiment")
    if any(n <= 0 or n >= dense_count for n in budgets):
        raise ValueError("Need positive budgets below the dense anchor count")

    anchor = np.asarray(mop.pf(n_points=dense_count), dtype=float)
    ideal, nadir = anchor.min(axis=0), anchor.max(axis=0)
    anchor_hv = independent_hv(name, anchor, ideal, nadir)
    if anchor_hv <= 0:
        raise AssertionError("Expected strictly positive anchor HV")

    rows = []
    nested_values = []
    for n in budgets:
        direct = np.asarray(mop.pf(n_points=n), dtype=float)
        nested = anchor[:n]
        direct_hv = independent_hv(name, direct, ideal, nadir)
        nested_hv = independent_hv(name, nested, ideal, nadir)
        nested_values.append(nested_hv)

        if name == "DTLZ7":
            # Stratified deterministic draws are repeatable across budgets.
            # A larger study may use independent seeds; this pilot does not.
            np.testing.assert_array_equal(direct, nested)

        rows.append({
            "n": int(n),
            "direct_hv": direct_hv,
            "nested_hv": nested_hv,
            "anchor_hv": anchor_hv,
            "direct_relative_to_anchor": direct_hv / anchor_hv,
            "nested_relative_to_anchor": nested_hv / anchor_hv,
            "direct_gap_to_anchor": anchor_hv - direct_hv,
            "nested_gap_to_anchor": anchor_hv - nested_hv,
        })

    if np.any(np.diff(nested_values) < -1e-12):
        raise AssertionError("Nested fronts must have nondecreasing exact HV")

    if check_public:
        # Compare the independent geometric oracle to the *unchanged*
        # public exact backend under the same fixed reference box.
        selected = np.asarray(mop.pf(n_points=budgets[0]), dtype=float)
        public = float(mb.metrics.hv(
            selected, ref=anchor, scale="raw", mode="exact",
            progress=False
        ))
        np.testing.assert_allclose(
            public, rows[0]["direct_hv"], rtol=3e-8, atol=1e-11
        )

    return {
        "problem": name,
        "objectives": int(anchor.shape[1]),
        "anchor_points": int(dense_count),
        "normalization": "fixed dense-anchor ideal/nadir",
        "anchor_hv": anchor_hv,
        "nested_hv_nondecreasing": True,
        "budgets": rows,
    }


def main():
    print(json.dumps({
        "study": "issue89-fixed-geometry-GT-refinement",
        "scope": "finite-sample convergence diagnostic, not certified bound",
        "cases": [
            analyze_case("DTLZ7", check_public=True),
            analyze_case("DPF1", check_public=True),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
