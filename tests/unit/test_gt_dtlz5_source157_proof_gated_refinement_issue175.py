# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive proof-gated selection tests with zero production scalar solves."""
from types import SimpleNamespace

import pytest

from scripts.audit.dtlz5_source157_proof_gated_refinement import (
    select_proven_source_candidate,
)
from scripts.audit.dtlz5_row10_exact_dominance import SOURCE_ROW_10

# Separate archived output was recorded as hex in 2026-11-26 source CI.
CERTIFIED_HEX = (
    "0x1.5642f4bea20bep-3",
    "0x1.0000000000000p+0",
    "0x1.0000000000000p+0",
    "0x1.5c70cb8e4c2eap-1",
)
CERTIFIED = tuple(float.fromhex(t) for t in CERTIFIED_HEX)


def _fake_solver(weights, max_g, *, abs_tol, max_nodes):
    assert max_g == .25 and max_nodes == 30000
    return SimpleNamespace(nodes=1, abs_gap=abs_tol, tol=abs_tol)


def test_known_dominated_decision_rejected_before_independent_valid_one():
    seen = []
    def reconstruct(candidate, **kwargs):
        seen.append(candidate.tol)
        return SOURCE_ROW_10 if candidate.tol == 1e-7 else CERTIFIED
    result = select_proven_source_candidate(
        [1.0]*4, tolerances=(1e-7,1e-9,1e-11),
        solver=_fake_solver, reconstruct=reconstruct,
    )
    assert seen == [1e-7,1e-9]
    assert result["accepted_attempt_index"] == 1
    assert [r["status"] for r in result["attempts"]] == [
        "rigorously_dominated", "rigorously_global_decision_image",
    ]
    assert result["attempts"][1]["exact_pf_proof"] is not None
    assert result["whole_gt_membership"] == "unverified"


def test_no_support_proof_fails_closed_without_promoting_approximation():
    def reconstruct(candidate, **kwargs):
        return SOURCE_ROW_10
    out = select_proven_source_candidate(
        [1.0]*4,tolerances=(1e-7,1e-9),
        solver=_fake_solver,reconstruct=reconstruct,
    )
    assert out["accepted_attempt_index"] is None
    assert out["source_candidate_status"] == "unverified_no_accepted_candidate"
    assert all(x["proved_dominated"] for x in out["attempts"])


def test_off_support_unknown_never_promoted():
    unknown = (0.1, 0.999999999, 1.0, 0.68)
    out = select_proven_source_candidate(
        [1.0]*4,tolerances=(1e-9,),
        solver=_fake_solver,
        reconstruct=lambda candidate,**kwargs: unknown,
    )
    assert out["accepted_attempt_index"] is None
    assert out["attempts"][0]["status"] in ("unverified","rigorously_dominated")
    assert out["scientific_certification"] == "unverified"


@pytest.mark.parametrize("bad",[0,-1,1,float("inf"),float("nan")])
def test_invalid_tolerance_rejected_without_solver(bad):
    def prohibited(*args,**kwargs):
        raise AssertionError("No scalarizer call allowed")
    with pytest.raises(ValueError):
        select_proven_source_candidate(
            [1]*4,tolerances=(bad,),
            solver=prohibited,
        )
