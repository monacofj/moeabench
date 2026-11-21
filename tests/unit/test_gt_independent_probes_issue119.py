# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independently constructed analytical front-probe contracts (#119)."""
import numpy as np
import pytest

from scripts.audit.gt_independent_probes_issue119 import (
    SUPPORTED, finite_probe_gap, independent_decision_probe,
)
from scripts.audit.gt_postfix_quantitative_issue119 import make_problem


@pytest.mark.parametrize("problem", SUPPORTED)
def test_independent_probe_feasible_and_reproducible(problem):
    m, d = 3, 2
    x = independent_decision_probe(problem, m, d, 64, 11941)
    mop = make_problem(problem, m, d)
    assert x.shape == (64, mop.N)
    np.testing.assert_array_equal(
        x, independent_decision_probe(problem, m, d, 64, 11941))
    f = mop.evaluation(x)["F"]
    assert np.isfinite(f).all()
    assert f.shape == (64, m)
    report = finite_probe_gap(problem, 128, m, d, probes=64)
    assert report["finite_mean_gap"] >= 0
    assert report["finite_max_gap"] >= report["finite_mean_gap"]
    assert not report["certified_continuous_bound"]


@pytest.mark.parametrize("problem", ["DTLZ7", "DPF2"])
def test_probe_component_report_does_not_hide_missing_regions(problem):
    m, d = 5, 3
    result = finite_probe_gap(problem, 128, m, d, probes=128)
    assert len(result["by_component"]) == (16 if problem == "DTLZ7" else 4)
    assert sum(v["probes"] for v in result["by_component"].values()) == 128


def test_unverified_oracle_is_not_silently_used():
    with pytest.raises(ValueError, match="No independently validated probe"):
        finite_probe_gap("DTLZ8", 128)
