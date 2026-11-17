# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent small-number metric oracles and conventional H_rel contracts.

All expected values are derived from 2-D geometry, rather than from another
call into the MoeaBench implementation under test (ADRs 0031 and 0047).
"""

from math import sqrt

import numpy as np
import pytest

import moeabench as mb
from moeabench.core.run import Run


REFERENCE_A = np.array([[0.0, 1.0], [1.0, 0.0]])
REFERENCE_B = np.array([[0.2, 0.8], [0.8, 0.2]])
SMALL_FRONT = REFERENCE_B.copy()


def _hv(front, *, ref, scale):
    return mb.metrics.hv(front, ref=ref, scale=scale, mode="exact", progress=False)


def _run(history):
    history = [np.asarray(front, dtype=float) for front in history]
    empty = [np.empty((len(front), 0)) for front in history]
    return Run((history, empty, history[-1], history, empty, empty, empty))


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("gd", sqrt(0.08)),
        ("igd", sqrt(0.08)),
        ("gdplus", 0.2),
        ("igdplus", 0.2),
        ("emd", 0.2),
    ],
)
def test_metric_values_against_hand_computed_2d_geometry(metric, expected):
    # Both objectives span [0, 1] in REFERENCE_A, so normalized distances
    # equal the hand-computed original-coordinate Euclidean/plus distances.
    actual = getattr(mb.metrics, metric)(
        SMALL_FRONT, ref=REFERENCE_A, progress=False
    )
    assert float(actual) == pytest.approx(expected, abs=1e-10)


def test_raw_hv_against_rectangle_union_oracle():
    # In the fixed [0,1]^2 bounds, pymoo's HV reference point is (1.1, 1.1).
    # Two 0.9 x 0.3 rectangles with a 0.3 x 0.3 overlap:
    # 0.27 + 0.27 - 0.09 = 0.45.
    assert float(_hv(SMALL_FRONT, ref=REFERENCE_A, scale="raw")) == pytest.approx(
        0.45
    )


def test_relative_hv_uses_best_individual_reference_not_pooled_union():
    # Each reference is evaluated separately in bounds [0,1]^2:
    #   HV(REFERENCE_A) = 0.21
    #   HV(REFERENCE_B) = 0.45
    # A pooled reference would yield a different, larger denominator.
    # HV([[0.5, 0.5]]) = 0.6 x 0.6 = 0.36.
    evaluated = np.array([[0.5, 0.5]])
    result = _hv(evaluated, ref=[REFERENCE_A, REFERENCE_B], scale="rel")
    assert float(result) == pytest.approx(0.36 / 0.45, abs=1e-10)


def test_relative_hv_keeps_one_final_denominator_over_all_generations():
    early = np.array([[0.05, 0.05]])
    final = np.array([[0.5, 0.5]])
    trajectory = _run([early, final])

    result = _hv(trajectory, ref=[REFERENCE_A, REFERENCE_B], scale="rel")
    # Early 1.05 x 1.05 rectangle; final 0.6 x 0.6 rectangle.
    np.testing.assert_allclose(
        result.values[:, 0],
        [1.1025 / 0.45, 0.36 / 0.45],
        atol=1e-10,
        rtol=0,
    )
    assert result.values[0, 0] > 1.0  # ADR 0047 forbids clipping.


def test_relative_hv_self_reference_uses_best_final_run():
    run_a = _run([
        np.array([[0.1, 0.1]]),
        np.array([[0.5, 0.5]]),
    ])
    run_b = _run([
        REFERENCE_B,
        REFERENCE_B,
    ])
    raw = mb.metrics.hv([run_a, run_b], scale="raw", mode="exact", progress=False)
    relative = mb.metrics.hv(
        [run_a, run_b], scale="rel", mode="exact", progress=False
    )

    expected_denominator = np.max(raw.values[-1, :])
    assert expected_denominator > 0
    np.testing.assert_allclose(
        relative.values, raw.values / expected_denominator, atol=1e-10, rtol=0
    )
    assert np.max(relative.values[-1, :]) == pytest.approx(1.0, abs=1e-10)
