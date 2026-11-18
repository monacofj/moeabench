# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Boundary guards for ADR 0021's float64 numerical-integrity contract."""

from io import StringIO

import numpy as np
import pandas as pd
import pytest

import moeabench as mb
from moeabench.core.run import Population, Run
from moeabench.metrics.evaluator import MetricMatrix


def test_public_population_promotes_numeric_inputs_to_float64():
    objectives = np.array([[0.125, 0.875]], dtype=np.float32)
    variables = np.array([[0.5, 0.25]], dtype=np.float32)
    original_objectives = objectives.copy()
    population = Population(objectives, variables)

    assert population.objectives.dtype == np.dtype("float64")
    assert population.variables.dtype == np.dtype("float64")
    np.testing.assert_allclose(population.objectives, original_objectives, rtol=0, atol=0)
    assert objectives.dtype == np.float32  # No mutation of the caller's data.


def test_run_histories_promote_numeric_inputs_to_float64():
    front = np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float32)
    variables = np.array([[0.1], [0.9]], dtype=np.float32)
    payload = (
        [front],
        [variables],
        front,
        [front],
        [variables],
        [np.empty((0, 2), dtype=np.float32)],
        [np.empty((0, 1), dtype=np.float32)],
    )
    run = Run(payload)

    for kind in ("f", "x", "nd", "nd_x", "dom", "dom_x"):
        assert run.history(kind)[0].dtype == np.dtype("float64")
    assert run.pop().objectives.dtype == np.dtype("float64")
    assert run.pop().variables.dtype == np.dtype("float64")


def test_metric_matrix_promotes_single_precision_inputs_to_float64():
    result = MetricMatrix(np.array([0.125, 0.25], dtype=np.float32))
    assert result.values.dtype == np.dtype("float64")


def test_mop_gt_calibration_promotes_external_single_precision(tmp_path):
    mop = mb.mops.DTLZ2(M=2)
    external = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert mop.calibrate(source_gt=external, source_baseline=tmp_path / "gt.json")
    assert mop.calibrated_gt.dtype == np.dtype("float64")


def test_pandas_csv_objective_import_preserves_float64_precision():
    # Probe the published precision policy without a heavy experiment.
    table = pd.read_csv(StringIO("f1,f2\n0.123456789012345,0.987654321098765\n"))
    assert table["f1"].dtype == np.dtype("float64")
    assert table["f2"].dtype == np.dtype("float64")
    assert table.iloc[0, 0] == pytest.approx(0.123456789012345, abs=1e-15)
