# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive exact dominance tests for archived original DTLZ5 source row 10."""
from copy import deepcopy
from fractions import Fraction
import pytest

from scripts.audit.dtlz5_row10_exact_dominance import (
    SOURCE_ROW_10, archived_row10_witness,
    certify_strict_dtlz5_m4_dominator,
    replay_strict_dtlz5_m4_dominator,
)


def test_archived_post_210_source_row10_has_rigorous_strict_dominator():
    source = SOURCE_ROW_10
    witness = archived_row10_witness()
    assert all(0 <= q <= 1 for q in witness)
    assert witness[1] < 1  # No snapping to an exact angular support face.
    assert witness[1] > Fraction.from_float(source[1])
    evidence = certify_strict_dtlz5_m4_dominator(source, witness)
    assert evidence["conclusion"] == "rigorously_strictly_dominated"
    assert all(
        int(m["numerator_hex"], 16) > 0
        for m in evidence["strict_objective_margins"]
    )
    assert replay_strict_dtlz5_m4_dominator(source, witness, evidence)
    assert evidence["scientific_certification"] == "unverified"


def test_forged_proof_and_original_inputs_do_not_replay():
    source = SOURCE_ROW_10
    witness = archived_row10_witness()
    proof = certify_strict_dtlz5_m4_dominator(source, witness)
    altered = deepcopy(proof)
    altered["strict_objective_margins"][0]["numerator_hex"] = "0x0"
    assert not replay_strict_dtlz5_m4_dominator(source, witness, altered)
    changed = list(source)
    changed[0] = 0.2
    assert not replay_strict_dtlz5_m4_dominator(changed, witness, proof)


def test_unimproved_candidate_abstains_and_fails_closed():
    same = certify_strict_dtlz5_m4_dominator(
        SOURCE_ROW_10, SOURCE_ROW_10,
    )
    assert same["conclusion"] == "unverified"
    assert not replay_strict_dtlz5_m4_dominator(
        SOURCE_ROW_10, SOURCE_ROW_10, same,
    )


@pytest.mark.parametrize("bad", [
    [0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1.1],
    [0, -1, 0, 0],
])
def test_reject_invalid_decision_vectors(bad):
    with pytest.raises(ValueError):
        certify_strict_dtlz5_m4_dominator(bad, archived_row10_witness())


def test_no_production_evaluator_sampler_or_solver(monkeypatch):
    from moeabench.mops import DTLZ5
    def fail(*args, **kwargs):
        raise AssertionError("Exact proof must be independent of production")
    monkeypatch.setattr(DTLZ5, "evaluation", fail)
    monkeypatch.setattr(DTLZ5, "ps", fail)
    evidence = certify_strict_dtlz5_m4_dominator(
        SOURCE_ROW_10, archived_row10_witness(),
    )
    assert evidence["conclusion"] == "rigorously_strictly_dominated"
