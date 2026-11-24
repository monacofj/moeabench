# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New tests for rational directional search; no existing regressions edited."""
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from scripts.audit.dtlz56_rational_tangent_search import (
    search_rational_tangent_witness,
    replay_rational_tangent_search,
)


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("positions", [
    [Q(1, 2), Q(1), Q(1)],
    [Q(1, 2), Q(0), Q(0)],
    [Q(1, 2), Q(0), Q(1, 2)],
])
def test_boundary_proof_found_by_deterministic_single_direction(problem, positions):
    options = dict(g=Q(1, 2),problem=problem,k=4,
                   max_candidates=1,pi_terms=6,trig_pairs=2)
    result = search_rational_tangent_witness(positions,**options)
    assert result.outcome == "rigorous_local_dominator_found"
    assert result.candidates_examined == 1
    assert result.derivative_certificate["conclusion"] == "rigorous_local_dominator"
    assert replay_rational_tangent_search(positions,result.to_dict(),
                                          g=options["g"],problem=problem,k=4,
                                          pi_terms=6,trig_pairs=2)


def test_unresolved_search_cannot_become_positive_certificate():
    # With theta1=0 and restricted first direction the only attempted
    # velocity cannot strictly decrease all objectives at first order.
    positions = [Q(0),Q(0),Q(1)]
    result = search_rational_tangent_witness(
        positions,g=Q(1,2),problem="DTLZ5",k=4,max_candidates=1,
    )
    assert result.outcome == "unverified"
    assert result.candidates_examined == 1
    assert result.velocity is None
    assert result.derivative_certificate is None
    assert not replay_rational_tangent_search(
        positions,result.to_dict(),g=Q(1,2),problem="DTLZ5",k=4,
    )


def test_tampering_and_wrong_input_are_rejected():
    positions = [Q(1,2),Q(1),Q(1)]
    options = dict(g=Q(1,2),problem="DTLZ5",k=4,max_candidates=1)
    record = search_rational_tangent_witness(
        positions,**options).to_dict()
    tampered = deepcopy(record)
    tampered["candidates_examined"] = 0
    assert not replay_rational_tangent_search(
        positions,tampered,g=Q(1,2),problem="DTLZ5",k=4,
    )
    tampered = deepcopy(record)
    tampered["derivative_certificate"]["derivative_intervals"][0]["lower"]["numerator_hex"] = "0x0"
    assert not replay_rational_tangent_search(
        positions,tampered,g=Q(1,2),problem="DTLZ5",k=4,
    )
    assert not replay_rational_tangent_search(
        [Q(1,3),Q(1),Q(1)],record,g=Q(1,2),problem="DTLZ5",k=4,
    )


@pytest.mark.parametrize("limit", [0, -1, 129, True, 1.5])
def test_candidate_budget_validation(limit):
    with pytest.raises(ValueError,match="max_candidates"):
        search_rational_tangent_witness(
            [Q(1,2),Q(1),Q(1)],g=Q(1,2),problem="DTLZ5",
            k=4,max_candidates=limit,
        )


def test_invalid_problem_is_rejected_even_before_search():
    with pytest.raises(ValueError):
        search_rational_tangent_witness(
            [Q(1,2),Q(1),Q(1)],g=Q(1,2),problem="DTLZ1",
            k=4,max_candidates=1,
        )
