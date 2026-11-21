# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract tests for a read-only issue #119 quantitative evidence collector."""
import numpy as np
import pytest

from scripts.audit.gt_postfix_quantitative_issue119 import (
    PROBLEMS, component_spacing, geometry_residual, region_counts, sample,
)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_postfix_report_reproduces_and_has_finite_objectives(name):
    result = sample(name, 128, m=3, d=2)
    assert result["problem"] == name
    assert result["budget_actual"] == 128
    assert result["unique_objectives"] > 0
    assert result["ps_reproducible"]
    assert result["elapsed_seconds"] >= 0
    assert len(result["objective_sha256"]) == 64
    if name in ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4",
                "DTLZ8", "DTLZ9"):
        assert result["geometry_residual"] < 1e-10


@pytest.mark.parametrize("name,m,d,components", [
    ("DTLZ7", 3, 2, 4),
    ("DTLZ7", 5, 2, 16),
    ("DPF2", 5, 3, 4),
])
def test_postfix_individual_disconnected_components(name, m, d, components):
    result = sample(name, 128, m=m, d=d)
    occupancy = result["region_occupancy"]
    assert occupancy["occupied"] == components
    assert occupancy["invalid_record_rows"] == 0
    assert len(result["component_spacing"]) == components
    assert sum(x["count"] for x in result["component_spacing"].values()) == 128


def test_postfix_dtlz8_component_counts_are_not_pooled():
    result = sample("DTLZ8", 128, m=4)
    assert len(result["component_spacing"]) == 4
    assert sum(x["count"] for x in result["component_spacing"].values()) == 128


def test_postfix_analytical_geometry_rejects_bad_simplex():
    good = np.array([[0.2, 0.3]])
    bad = np.array([[0.2, 0.4]])
    assert geometry_residual("DTLZ1", good) == 0
    assert geometry_residual("DTLZ1", bad) > 0.09
