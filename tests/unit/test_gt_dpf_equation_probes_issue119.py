# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent published-equation probes, no old tests changed."""
import numpy as np

from scripts.audit.gt_dpf_equation_probes_issue119 import (
    dpf3_threshold_probe, dpf5_threshold_probe,
)


def test_single_eta_has_no_ordering_question():
    row = dpf3_threshold_probe(3, 64)
    assert row["source_order_strictly_increasing"]
    assert np.isclose(row["actual_vs_sorted_max_abs"], 0, atol=1e-12)


def test_m5_evaluator_uses_published_sorted_thresholds():
    row = dpf3_threshold_probe(5, 128)
    assert np.all(np.diff(row["eta_sorted"]) > 0)
    assert row["actual_vs_sorted_max_abs"] < 1e-12


def test_dpf5_equality_belongs_to_published_d_branch():
    row = dpf5_threshold_probe(5)
    assert row["current_at_equality"] == row["published_at_equality"] == "gamma/D"
    assert row["equality_minus_right_max_abs"] < row["equality_minus_left_max_abs"]
