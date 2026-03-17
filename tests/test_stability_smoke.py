# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fast numerical stability smoke tests for analytical outputs and solver fronts."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import moeabench as mb
from moeabench.diagnostics import auditor


DATA_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration_reference_audit_v0.13.2.json")
TARGETS_JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration_reference_targets_v0.13.2.json")
FRONTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "stability_reference_fronts.json")
GT_JSON_PATH = os.path.join(os.path.dirname(__file__), "stability_reference_gt.json")

with open(DATA_JSON_PATH, "r") as f:
    CALIBRATION_REF_DATA = json.load(f)

with open(TARGETS_JSON_PATH, "r") as f:
    CALIBRATION_REF_TARGETS = json.load(f)

with open(FRONTS_JSON_PATH, "r") as f:
    STABILITY_FRONTS = json.load(f)

with open(GT_JSON_PATH, "r") as f:
    STABILITY_GT = json.load(f)


SMOKE_ANALYTICAL_CASES = [
    ("DTLZ2", "NSGA3"),
    ("DPF1", "MOEAD"),
]
SMOKE_GT_CASES = [
    "DTLZ8",
    "DTLZ9",
]
RNG_SIGNATURE_SEED = 7


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


def test_rng_backend_signature_python_random():
    """Sentinel for Python RNG drift; stability failures may be environmental if this breaks."""
    rng = random.Random(RNG_SIGNATURE_SEED)
    observed = [rng.random() for _ in range(5)]
    expected = [
        0.32383276483316237,
        0.15084917392450192,
        0.6509344730398537,
        0.07243628666754276,
        0.5358820043066892,
    ]
    assert observed == expected, (
        "Python stdlib RNG signature drift detected. "
        "Check Python/runtime version drift first; downstream MoeaBench stability failures "
        "may be environmental rather than a regression in the codebase."
    )


def test_rng_backend_signature_numpy_randomstate():
    """Sentinel for NumPy RandomState drift; core references depend on this contract."""
    rng = np.random.RandomState(RNG_SIGNATURE_SEED)
    observed_random = rng.random_sample(5).tolist()
    observed_int = np.random.RandomState(RNG_SIGNATURE_SEED).randint(0, 1000, size=5).tolist()
    expected_random = [
        0.07630828937395717,
        0.7799187922401146,
        0.4384092314408935,
        0.7234651778309412,
        0.9779895119966027,
    ]
    expected_int = [175, 196, 537, 502, 579]
    assert observed_random == expected_random, (
        "NumPy RandomState random_sample signature drift detected. "
        "Check NumPy/runtime version drift first; downstream MoeaBench stability failures "
        "may be environmental rather than a regression in the codebase."
    )
    assert observed_int == expected_int, (
        "NumPy RandomState randint signature drift detected. "
        "Check NumPy/runtime version drift first; downstream MoeaBench stability failures "
        "may be environmental rather than a regression in the codebase."
    )


def test_rng_backend_signature_numpy_default_rng():
    """Sentinel for NumPy Generator drift in newer calibration/reference paths."""
    rng = np.random.default_rng(RNG_SIGNATURE_SEED)
    observed_random = rng.random(5).tolist()
    observed_normal = np.random.default_rng(RNG_SIGNATURE_SEED).normal(size=5).tolist()
    observed_int = np.random.default_rng(RNG_SIGNATURE_SEED).integers(0, 1000, size=5).tolist()
    expected_random = [
        0.625095466604667,
        0.8972138009695755,
        0.7756856902451935,
        0.22520718999059186,
        0.30016628491122543,
    ]
    expected_normal = [
        0.0012301533574825742,
        0.2987455375084699,
        -0.2741378553622176,
        -0.8905918387572742,
        -0.45467078517172255,
    ]
    expected_int = [944, 625, 684, 897, 578]
    assert observed_random == expected_random, (
        "NumPy default_rng random signature drift detected. "
        "Check NumPy/runtime version drift first; downstream MoeaBench stability failures "
        "may be environmental rather than a regression in the codebase."
    )
    assert observed_normal == expected_normal, (
        "NumPy default_rng normal signature drift detected. "
        "Check NumPy/runtime version drift first; downstream MoeaBench stability failures "
        "may be environmental rather than a regression in the codebase."
    )
    assert observed_int == expected_int, (
        "NumPy default_rng integers signature drift detected. "
        "Check NumPy/runtime version drift first; downstream MoeaBench stability failures "
        "may be environmental rather than a regression in the codebase."
    )


def test_experiment_seed_schedule_signature(monkeypatch):
    """Experiment repeats must advance deterministically as base_seed + i."""
    exp = mb.experiment(mop=_build_problem("DTLZ2"))
    exp.moea = mb.moeas.NSGA2deap(population=4, generations=1, seed=123)

    used_seeds = []

    def fake_execute_run(moea, mop, seed, index):
        used_seeds.append(seed)
        return ([], [], np.zeros((0, 2)), [], [], [], []), seed

    monkeypatch.setattr(exp, "_execute_run", fake_execute_run)

    exp.run(repeat=3, silent=True)

    assert used_seeds == [123, 124, 125], (
        "Experiment seed schedule drift detected. "
        "Repeated runs must follow base_seed + i for deterministic replay and stability certification."
    )


@pytest.mark.parametrize(
    "problem_name, alg_name",
    SMOKE_ANALYTICAL_CASES,
    ids=[f"{alg_name}x{problem_name}" for problem_name, alg_name in SMOKE_ANALYTICAL_CASES],
)
def test_smoke_metrics_reproducibility(problem_name, alg_name):
    p_data = CALIBRATION_REF_DATA["problems"][problem_name]
    gt_points = np.array(p_data["gt_points"])
    final_front = np.array(p_data["algorithms"][alg_name]["final_front"])
    target_key = f"{problem_name}/{alg_name}"
    target_entry = CALIBRATION_REF_TARGETS[target_key]

    report = auditor.audit(final_front, ground_truth=gt_points, problem=problem_name)
    q_audit = report.q_audit_res
    fair_audit = report.fair_audit_res

    assert q_audit is not None
    assert fair_audit is not None

    for key, target_val in target_entry["q_scores"].items():
        if key in q_audit.scores:
            assert float(q_audit.scores[key].value) == pytest.approx(target_val, abs=1e-6, nan_ok=True)

    for key, target_val in target_entry["raw_metrics"].items():
        if key in fair_audit.metrics:
            assert float(fair_audit.metrics[key].value) == pytest.approx(target_val, abs=1e-6, nan_ok=True)


@pytest.mark.parametrize("problem_name", SMOKE_GT_CASES, ids=SMOKE_GT_CASES)
def test_smoke_gt_reproducibility(problem_name):
    expected = STABILITY_GT["smoke"][problem_name]
    forward_max, backward_max = _gt_geometry_signature(problem_name)
    assert forward_max <= expected["forward_max"] + 1e-12
    assert backward_max <= expected["backward_max"] + 1e-12


@pytest.mark.parametrize(
    "entry",
    STABILITY_FRONTS["smoke"],
    ids=lambda entry: f"{entry['algorithm']}x{entry['mop']}",
)
def test_smoke_front_reproducibility(entry):
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
