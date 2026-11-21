# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract tests for read-only DTLZ8 candidate-selection evidence."""
import pytest

from scripts.audit.gt_dtlz8_policy_issue119 import summarize


@pytest.mark.parametrize("m", [2, 3, 5, 10])
def test_policy_candidate_admissibility_and_component_specific_results(m):
    evidence = summarize(m, 128, probes=32)
    assert evidence["requested"] == 128
    assert evidence["M"] == m
    assert not evidence["production_policy_approved"]
    assert not evidence["canonical_mixed_dimensional_measure_defined"]
    assert not evidence["continuous_coverage_certified"]
    assert not evidence["incumbent_reused"]
    assert len(evidence["per_component"]) == (1 if m == 2 else m)
    for candidate in evidence["admissibility"].values():
        assert candidate["count"] == candidate["unique"] == 128
        assert candidate["max_pareto_residual"] < 1e-10
        assert candidate["max_constraint_violation"] < 1e-10
    assert evidence["finite_probe_max_gap_preference"] in (
        "lattice", "halton", "tie", "mixed_no_proven_winner")
    for component in evidence["per_component"].values():
        assert len(component["candidate"]["max_gap_by_offset"]) == 3
        assert len(component["lattice"]["max_gap_by_offset"]) == 3


def test_policy_needs_independent_probe_offsets():
    with pytest.raises(ValueError, match="distinct"):
        summarize(3, 128, offsets=(11941, 11941, 11942), probes=16)


def test_policy_refuses_untested_geometry():
    with pytest.raises(ValueError, match="covers"):
        summarize(4, 128, probes=16)


def test_policy_does_not_silently_allow_too_small_budget():
    with pytest.raises(ValueError, match="Budget"):
        summarize(10, 10, probes=16)
