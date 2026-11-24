# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent additional partition/replay tests; preserve existing regressions."""
from copy import deepcopy
from fractions import Fraction
from math import cos, pi, sin

import pytest

from scripts.audit.dtlz56_rational_interval import (
    _as_exact_fraction, rigorously_exclude_g_interval,
)
from scripts.audit.dtlz56_rational_partition import (
    partition_excluded_g_range, replay_partition_proof,
)


def _independent_feasible_point(g):
    r = 1+g
    a = pi/(4*r)
    angle = pi/4
    return [
        r*cos(angle)*cos(a),
        r*cos(angle)*sin(a),
        r*sin(angle),
    ]


def test_rational_input_is_not_float_rounded():
    x = Fraction(2**70+1, 2**70)
    assert _as_exact_fraction(x, name="x") == x
    assert _as_exact_fraction(x, name="x") != Fraction.from_float(float(x))
    result = rigorously_exclude_g_interval(
        [0,0,0],g_lower=Fraction(1, 7),g_upper=Fraction(2,7),
        problem="DTLZ6",k=4,pi_terms=4,sine_pairs=1,
    )
    assert result.g_lower == "1/7"
    assert result.g_upper == "2/7"


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [2,3,4,5,10])
def test_partition_proves_absence_on_clearly_impossible_caps(problem,m):
    audit = partition_excluded_g_range(
        [0]*m,g_upper=Fraction(1,2),problem=problem,k=4,
        max_nodes=7,max_depth=3,pi_terms=4,sine_pairs=1,
    )
    assert audit.outcome == "rigorously_excluded_closed_range"
    assert audit.excluded_leaves == 1
    assert audit.unresolved_leaves == 0
    assert replay_partition_proof(
        [0]*m,audit.to_dict(),problem=problem,k=4,
        pi_terms=4,sine_pairs=1,
    )


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
def test_no_certificate_when_range_contains_feasible_candidate(problem):
    y = _independent_feasible_point(.25)
    audit = partition_excluded_g_range(
        y,g_upper=Fraction(1,2),problem=problem,k=4,
        max_nodes=15,max_depth=3,pi_terms=4,sine_pairs=1,
    )
    assert audit.outcome == "unverified"
    assert audit.unresolved_leaves > 0
    assert not replay_partition_proof(
        y,audit,problem=problem,k=4,pi_terms=4,sine_pairs=1,
    )


def test_budget_exhaustion_never_makes_full_certification():
    y = _independent_feasible_point(.25)
    audit = partition_excluded_g_range(
        y,g_upper=Fraction(1,2),problem="DTLZ6",k=4,
        max_nodes=1,max_depth=5,pi_terms=4,sine_pairs=1,
    )
    assert audit.explored_nodes == 1
    assert audit.outcome == "unverified"
    assert any(x["status"] == "unresolved_budget" for x in audit.leaves)
    assert not replay_partition_proof(
        y,audit,problem="DTLZ6",k=4,pi_terms=4,sine_pairs=1,
    )


def test_replay_rejects_omitted_leaf_and_tampered_margin():
    audit = partition_excluded_g_range(
        [0,0,0,0],g_upper=Fraction(1,2),problem="DTLZ5",k=4,
        max_nodes=3,max_depth=1,pi_terms=4,sine_pairs=1,
    ).to_dict()
    assert replay_partition_proof(
        [0,0,0,0],audit,problem="DTLZ5",k=4,
        pi_terms=4,sine_pairs=1,
    )
    damaged = deepcopy(audit)
    damaged["leaves"][0]["certificate"]["strict_margin"]["numerator_hex"] = "0x0"
    assert not replay_partition_proof(
        [0,0,0,0],damaged,problem="DTLZ5",k=4,
        pi_terms=4,sine_pairs=1,
    )
    damaged = deepcopy(audit)
    damaged["leaves"] = []
    assert not replay_partition_proof(
        [0,0,0,0],damaged,problem="DTLZ5",k=4,
        pi_terms=4,sine_pairs=1,
    )


def test_nonzero_rational_lower_endpoint_round_trips():
    low = Fraction(2**70+1,2**70)
    upper = low+Fraction(1,2**60)
    audit = partition_excluded_g_range(
        [0,0],g_lower=low,g_upper=upper,problem="DTLZ6",k=4,
        max_nodes=3,max_depth=1,pi_terms=4,sine_pairs=1,
    )
    assert audit.outcome == "rigorously_excluded_closed_range"
    assert replay_partition_proof(
        [0,0],audit,problem="DTLZ6",k=4,
        pi_terms=4,sine_pairs=1,
    )


@pytest.mark.parametrize("opts", [
    {"max_nodes": 0}, {"max_nodes": True},
    {"max_depth": -1}, {"max_depth": 17},
])
def test_invalid_budget_is_rejected(opts):
    with pytest.raises(ValueError):
        partition_excluded_g_range(
            [0,0,0],g_upper=Fraction(1,2),problem="DTLZ6",k=4,
            pi_terms=4,sine_pairs=1,**opts,
        )
