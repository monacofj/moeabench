# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference-resolution IGD/IGD+ research controls (issue #119)."""
import numpy as np
import pytest

from scripts.audit.gt_reference_sensitivity_issue119 import (
    examine, raw_igd_and_igd_plus,
)


def test_raw_igd_plus_matches_positive_part_definition():
    reference = np.array([[0.2, 0.3]])
    worse = np.array([[0.4, 0.5]])
    better = np.array([[0.1, 0.2]])
    np.testing.assert_allclose(
        raw_igd_and_igd_plus(reference, worse)["raw_igd_plus"],
        np.sqrt(0.2**2 + 0.2**2), rtol=0, atol=1e-15)
    assert raw_igd_and_igd_plus(reference, better)["raw_igd_plus"] == 0
    assert raw_igd_and_igd_plus(reference, better)["raw_igd"] > 0


@pytest.mark.parametrize("name", ["DTLZ1", "DTLZ2", "DTLZ7", "DTLZ9",
                                   "DPF1", "DPF2"])
def test_reference_change_uses_the_same_fixed_population(name):
    report = examine(name, m=3, d=2, seed=11941,
                     population_size=32, budgets=(128, 512))
    assert report["fixed_population_size"] == 32
    assert report["population_type"] == "independent analytical synthetic probes"
    assert not report["production_normalization"]
    assert not report["actual_moea_performance_test"]
    assert not report["hv_test"]
    assert len(report["reference_scores"]) == 2
    for score in report["reference_scores"]:
        assert score["raw_igd"] >= score["raw_igd_plus"] >= 0


def test_independent_reference_sensitivity_rejects_unvalidated_problem():
    with pytest.raises(ValueError, match="Missing"):
        examine("DTLZ8")
