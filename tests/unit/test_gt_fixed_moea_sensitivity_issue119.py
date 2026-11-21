# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive contract tests for issue-119 frozen-MOEA sensitivity evidence."""
import numpy as np

from scripts.audit.gt_fixed_moea_sensitivity_issue119 import (
    analyze, evaluate_one, metric_snapshot, run_frozen_population,
)


def test_public_metrics_respond_to_fixed_reference_changes():
    front = np.array([[0.4, 0.8, 0.55], [0.8, 0.4, 0.65]])
    a = np.array([[0.3, 0.9, 0.5], [0.9, 0.3, 0.7]])
    b = np.array([[0.2, 1.0, 0.45], [1.0, 0.2, 0.8]])
    left = metric_snapshot(front, a)
    right = metric_snapshot(front, b)
    assert set(left) == set(right)
    assert all(np.isfinite(list(left.values())))
    assert left["raw_igd"] != right["raw_igd"]
    assert left["public_igd_plus"] != right["public_igd_plus"]


def test_one_actual_moea_run_is_evaluated_without_rerunning():
    record = evaluate_one("DTLZ2", "NSGA2", 11941,
                          population=16, generations=4,
                          reference_budgets=(32, 64))
    assert record["seed"] == 11941
    assert len(record["final_front_sha256"]) == 64
    assert len(record["reference_evaluations"]) == 2
    assert [r["reference_budget"] for r in
            record["reference_evaluations"]] == [32, 64]
    assert all(len(r["reference_sha256"]) == 64
               for r in record["reference_evaluations"])


def test_rank_analysis_flags_only_strict_sign_reversal():
    def case(algorithm, a, b):
        return {
            "problem": "DTLZ2", "algorithm": algorithm, "seed": 11941,
            "reference_evaluations": [
                {"reference_budget": 128, "scores": {
                    "public_igd": a, "public_igd_plus": a,
                    "public_hv_raw_reference_context": a}},
                {"reference_budget": 512, "scores": {
                    "public_igd": b, "public_igd_plus": b,
                    "public_hv_raw_reference_context": b}},
            ],
        }
    ordered = analyze([case("NSGA2", 0.1, 0.4),
                       case("SPEA2", 0.2, 0.3)])
    assert len(ordered) == 3
    assert all(x["ordering_reversed"] for x in ordered)
