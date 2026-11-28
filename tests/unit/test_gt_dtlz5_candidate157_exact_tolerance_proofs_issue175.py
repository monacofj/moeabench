# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive exact-decision proof audit of source-bound tolerance observations.

Source declarations are LOSSLESS float.hex strings captured by the
2026-11-26 CI Sobol-157 sweep, never rounded displayed decimals.
"""
from fractions import Fraction
from scripts.audit.dtlz56_tight_sine_global_pf import (
    certify_tight_sine_global_pf,
    replay_tight_sine_global_pf,
)
from scripts.audit.dtlz56_exact_same_ray import (
    construct_same_ray_witness,
)
from scripts.audit.dtlz5_row10_exact_dominance import (
    SOURCE_ROW_10, archived_row10_witness,
    certify_strict_dtlz5_m4_dominator,
)

AT_1E9 = (
    "0x1.5642f4bea20bep-3",
    "0x1.0000000000000p+0",
    "0x1.0000000000000p+0",
    "0x1.5c70cb8e4c2eap-1",
)
AT_1E11 = (
    "0x1.5642f4bb667ffp-3",
    "0x1.ffffffffe5810p-1",
    "0x1.0000000000000p+0",
    "0x1.5c70cb975e46ep-1",
)


def _decision(hex_values):
    return tuple(float.fromhex(value) for value in hex_values)


def _exact_g(decision):
    return (Fraction.from_float(decision[3]) - Fraction(1, 2))**2


def test_original_source_at_1e7_is_exactly_strictly_dominated():
    result = certify_strict_dtlz5_m4_dominator(
        SOURCE_ROW_10, archived_row10_witness(),
    )
    assert result["conclusion"] == "rigorously_strictly_dominated"


def test_1e9_source_decision_has_exact_global_pf_support_certificate():
    decision = _decision(AT_1E9)
    assert decision[1:3] == (1.0, 1.0)
    result = certify_tight_sine_global_pf(
        decision[:3],g=_exact_g(decision),problem="DTLZ5",k=1,
    )
    assert result.conclusion == "rigorously_global_pf_member"
    assert replay_tight_sine_global_pf(
        decision[:3],result.to_dict(),g=_exact_g(decision),
        problem="DTLZ5",k=1,
    )
    assert result.to_dict()["scientific_certification"] == "unverified"


def test_1e11_source_decision_not_promoted_from_tighter_tolerance():
    decision = _decision(AT_1E11)
    assert decision[1] < 1 and decision[2] == 1
    evidence = certify_tight_sine_global_pf(
        decision[:3],g=_exact_g(decision),problem="DTLZ5",k=1,
    )
    assert evidence.conclusion == "unverified"
    assert evidence.supporting_objective is None
    same_ray = construct_same_ray_witness(
        decision[:3],g=_exact_g(decision),problem="DTLZ5",k=1,
    )
    assert same_ray.conclusion == "no_strict_same_ray_reduction"
