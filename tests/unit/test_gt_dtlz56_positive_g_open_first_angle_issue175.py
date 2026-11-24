# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New global positive-membership tests for NONZERO original first angle.

No existing regression tests or numerical baselines are changed.
"""
from copy import deepcopy
from fractions import Fraction as Q
from math import cos, pi, sin, sqrt

import pytest

from scripts.audit.dtlz56_positive_g_open_first_angle import (
    certify_open_first_angle_global_pf,
    replay_open_first_angle_global_pf,
)


def _supported_positions(m, objective, first):
    """Construct boundary support using original angular indexing."""
    u = [Q(1, 3)] * (m - 1)
    u[0] = first
    if objective == 1:
        for i in range(1, m - 1):
            u[i] = Q(1)
    else:
        sine_position = m - objective
        for i in range(1, sine_position):
            u[i] = Q(1)
        u[sine_position] = Q(0)
    return u


def _independent_original_objectives(u, g):
    """Original spherical equations, independent of production evaluation."""
    m = len(u) + 1
    r = 1 + float(g)
    theta = [pi * float(u[0]) / 2]
    theta.extend(
        pi * (1 + 2 * float(g) * float(v)) / (4 * r)
        for v in u[1:]
    )
    output = []
    for j in range(m):
        value = r
        for i in range(m - j - 1):
            value *= cos(theta[i])
        if j:
            value *= sin(theta[m - j - 1])
        output.append(value)
    return output


@pytest.mark.parametrize("problem", ["DTLZ5", "DTLZ6"])
@pytest.mark.parametrize("m", [4, 5, 7, 10])
@pytest.mark.parametrize("g", [Q(1, 8), Q(1, 2), Q(1)])
def test_nonzero_first_angle_global_pf_every_support(problem, m, g):
    for j in range(1, m - 1):
        u = _supported_positions(m, j, Q(1, 16))
        certificate = certify_open_first_angle_global_pf(
            u, g=g, problem=problem, k=4,
        )
        assert certificate.conclusion == "rigorously_global_pf_member"
        assert certificate.sine_power >= 2
        assert certificate.supporting_objective == j
        assert certificate.to_dict()["scientific_certification"] == "unverified"
        assert replay_open_first_angle_global_pf(
            u, certificate.to_dict(), g=g, problem=problem, k=4,
        )
        y = _independent_original_objectives(u, g)
        r = 1 + float(g)
        c = r * sin(pi * float(u[0]) / 2)
        power = certificate.sine_power
        assert 0 < y[-1] < 1
        expected = sqrt(r*r-c*c)*sin(pi/(4*r))**power
        assert abs(y[j-1]-expected) < 1e-12
        assert y[-1] == pytest.approx(c, abs=1e-14)


def test_open_set_in_first_angle_with_arbitrarily_small_positive_g():
    g = Q(1, 2**220)
    u = _supported_positions(5, 3, Q(1, 9))
    certificate = certify_open_first_angle_global_pf(
        u, g=g, problem="DTLZ6", k=4,
    )
    assert certificate.conclusion == "rigorously_global_pf_member"
    assert replay_open_first_angle_global_pf(
        u, certificate, g=g, problem="DTLZ6", k=4,
    )


@pytest.mark.parametrize("m,objective", [(4, 1), (4, 2), (7, 1), (7, 5)])
def test_monotone_support_bound_over_entire_smaller_g_domain(m, objective):
    u = _supported_positions(m, objective, Q(1, 16))
    g = 0.5
    r = 1+g
    c = r*sin(pi*float(u[0])/2)
    p = m-2 if objective == 1 else m-objective
    last_value = sqrt(r*r-c*c)*sin(pi/(4*r))**p
    for gp in (0, 0.01, 0.1, 0.2, 0.3, 0.45, 0.499):
        rp = 1+gp
        lower = sqrt(rp*rp-c*c)*sin(pi/(4*rp))**p
        assert lower > last_value


@pytest.mark.parametrize("positions,g", [
    ([Q(1), Q(1), Q(0)], Q(1, 2)),
    ([Q(3, 4), Q(1), Q(0)], Q(1)),
    ([Q(1, 2), Q(1), Q(0)], Q(1)),
])
def test_unsafe_large_last_objective_must_abstain(positions, g):
    certificate = certify_open_first_angle_global_pf(
        positions, g=g, problem="DTLZ5", k=4,
    )
    assert certificate.conclusion == "unverified"
    assert not replay_open_first_angle_global_pf(
        positions, certificate, g=g, problem="DTLZ5", k=4,
    )


def test_missing_inner_boundary_support_must_abstain():
    u = [Q(1, 16), Q(1, 2), Q(1, 2)]
    certificate = certify_open_first_angle_global_pf(
        u, g=Q(1, 2), problem="DTLZ5", k=4,
    )
    assert certificate.conclusion == "unverified"
    assert certificate.supporting_objective is None


def test_exact_rational_pi_bounds_and_strict_margin_are_recorded():
    u = _supported_positions(4, 2, Q(1, 16))
    certificate = certify_open_first_angle_global_pf(
        u, g=Q(1, 2), problem="DTLZ5", k=4,
    )
    lower_margin = certificate.strict_rational_margin
    margin = Q(int(lower_margin["numerator_hex"], 16),
               int(lower_margin["denominator_hex"], 16))
    assert margin > 0
    import json
    json.dumps(certificate.to_dict(), allow_nan=False)


def test_forged_proof_cannot_be_replayed():
    u = _supported_positions(5, 2, Q(1, 16))
    args = dict(g=Q(1, 2), problem="DTLZ5", k=4)
    record = certify_open_first_angle_global_pf(u, **args).to_dict()
    assert replay_open_first_angle_global_pf(u, record, **args)
    mutated = deepcopy(record)
    mutated["strict_rational_margin"]["numerator_hex"] = "0x0"
    assert not replay_open_first_angle_global_pf(u, mutated, **args)
    assert not replay_open_first_angle_global_pf(
        u, record, g=Q(1, 3), problem="DTLZ5", k=4,
    )
    assert not replay_open_first_angle_global_pf(
        u, record, g=Q(1, 2), problem="DTLZ6", k=4,
    )


@pytest.mark.parametrize("u,g,problem,k,precision", [
    ([Q(0), Q(1)], Q(1, 2), "DTLZ5", 4, 8),
    ([Q(-1), Q(1), Q(0)], Q(1, 2), "DTLZ5", 4, 8),
    ([Q(1, 3), Q(1), Q(0)], Q(-1, 2), "DTLZ5", 4, 8),
    ([Q(1, 3), Q(1), Q(0)], Q(2), "DTLZ5", 4, 8),
    ([Q(1, 3), Q(1), Q(0)], Q(1, 2), "DTLZ1", 4, 8),
    ([Q(1, 3), Q(1), Q(0)], Q(1, 2), "DTLZ6", 0, 8),
    ([Q(1, 3), Q(1), Q(0)], Q(1, 2), "DTLZ6", 4, 1),
    ([Q(1, 3), Q(1), Q(0)], Q(1, 2), "DTLZ6", 4, 33),
])
def test_invalid_original_parameters_are_rejected(u, g, problem, k, precision):
    with pytest.raises(ValueError):
        certify_open_first_angle_global_pf(
            u, g=g, problem=problem, k=k, pi_terms=precision,
        )
