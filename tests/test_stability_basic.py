# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Basic numerical stability suite for analytical outputs and solver fronts."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import moeabench as mb


DATA_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration_reference_audit_v0.13.2.json")
FRONTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "stability_reference_fronts.json")
GT_JSON_PATH = os.path.join(os.path.dirname(__file__), "stability_reference_gt.json")

with open(DATA_JSON_PATH, "r") as f:
    CALIBRATION_REF_DATA = json.load(f)

with open(FRONTS_JSON_PATH, "r") as f:
    STABILITY_FRONTS = json.load(f)

with open(GT_JSON_PATH, "r") as f:
    STABILITY_GT = json.load(f)

GT_PROBLEMS = [
    "DTLZ8",
    "DTLZ9",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DPF5",
]


def _build_problem(mop_name: str):
    mop_cls = getattr(mb.mops, mop_name)
    try:
        return mop_cls(M=3)
    except TypeError:
        return mop_cls(M=3, D=2)


def _sort_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    order = np.lexsort(arr.T[::-1])
    return arr[order]


def _gt_geometry_signature(problem_name: str) -> tuple[float, float]:
    p_data = CALIBRATION_REF_DATA["problems"][problem_name]
    expected = np.asarray(p_data["gt_points"], dtype=float)
    mop = _build_problem(problem_name)
    observed = np.asarray(mop.pf(n_points=len(expected)), dtype=float)
    forward = cdist(observed, expected).min(axis=1)
    backward = cdist(expected, observed).min(axis=1)
    return float(np.max(forward)), float(np.max(backward))


@pytest.mark.parametrize("problem_name", GT_PROBLEMS, ids=GT_PROBLEMS)
def test_gt_reference_reproducibility(problem_name):
    # DTLZ2/3/4 have an analytical, continuous hyperspherical front.
    # Historical calibration points are samples from that surface, NOT
    # the canonical coordinates of the surface itself. A nearest-neighbor
    # Hausdorff distance between two independent samples is sensitive to
    # coverage and is not an appropriate numerical stability signature.
    if problem_name in {"DTLZ2", "DTLZ3", "DTLZ4"}:
        n_points = len(CALIBRATION_REF_DATA["problems"][problem_name]["gt_points"])
        mop = _build_problem(problem_name)
        observed = np.asarray(mop.pf(n_points=n_points), dtype=float)
        replay = np.asarray(mop.pf(n_points=n_points), dtype=float)

        # Identical inputs must replay identically, including their ordering.
        assert np.array_equal(observed, replay)
        assert observed.shape == (n_points, mop.M)
        assert np.isfinite(observed).all()
        # DTLZ2/3/4 Pareto fronts are the positive unit hypersphere.
        np.testing.assert_allclose(np.linalg.norm(observed, axis=1), 1.0,
                                   atol=1e-12, rtol=0)
        assert np.min(observed) >= -1e-12
        # The sampler explicitly includes the canonical axis extrema.
        np.testing.assert_allclose(observed[:mop.M], np.eye(mop.M),
                                   atol=1e-12, rtol=0)
        return

    # For other MOPs, retain the existing developer-certified geometric
    # sample baselines and thresholds unchanged.
    expected = STABILITY_GT["basic"][problem_name]
    forward_max, backward_max = _gt_geometry_signature(problem_name)
    assert forward_max <= expected["forward_max"] + 1e-12
    assert backward_max <= expected["backward_max"] + 1e-12


@pytest.mark.parametrize(
    "entry",
    STABILITY_FRONTS["basic"],
    ids=lambda entry: f"{entry['algorithm']}x{entry['mop']}",
)
def test_front_reproducibility(entry):
    mop = _build_problem(entry["mop"])
    alg = getattr(mb.moeas, entry["algorithm"])(
        seed=entry["seed"],
        population=entry["population"],
        generations=entry["generations"],
    )
    exp = mb.experiment(mop=mop)
    exp.moea = alg
    exp.run(repeat=1, silent=True)

    observed = _sort_rows(np.asarray(exp.front()))
    expected = _sort_rows(np.asarray(entry["front"], dtype=float))
    assert observed.shape == expected.shape
    assert np.allclose(observed, expected, atol=1e-12, rtol=1e-8)
