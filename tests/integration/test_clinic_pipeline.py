# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import moeabench as mb
import pytest


def test_clinic_pipeline(paired_experiments, canonical_gt):
    exp1, exp2 = paired_experiments

    diag1 = mb.clinic.audit(exp1)
    diag2 = mb.clinic.audit(exp2)
    close1 = mb.clinic.closeness(exp1, ref=canonical_gt)
    close2 = mb.clinic.closeness(exp2, ref=canonical_gt)

    assert diag1.q_audit_res is not None
    assert diag1.fair_audit_res is not None
    assert diag2.q_audit_res is not None
    assert close1.history_values is not None
    assert close2.raw_data is not None

    info = mb.system.info(show=False)
    assert "python_version" in info

    assert mb.view.radar(diag1, diag2, mode="static", show=False) is not None
    assert mb.view.ecdf(close1, mode="static", show=False) is not None
    assert mb.view.ecdf(close2, mode="static", show=False) is not None
    assert mb.view.density(close1, mode="static", show=False) is not None
    assert mb.view.history(close1, mode="static", show=False) is not None


def test_single_metric_api_matches_audit(paired_experiments):
    exp, _ = paired_experiments
    diagnostic = mb.clinic.audit(exp)

    fair_metrics = {
        "HEADWAY": mb.clinic.headway,
        "CLOSENESS": mb.clinic.closeness,
        "COVERAGE": mb.clinic.coverage,
        "GAP": mb.clinic.gap,
        "REGULARITY": mb.clinic.regularity,
        "BALANCE": mb.clinic.balance,
    }
    quality_scores = {
        "Q_HEADWAY": mb.clinic.q_headway,
        "Q_CLOSENESS": mb.clinic.q_closeness,
        "Q_COVERAGE": mb.clinic.q_coverage,
        "Q_GAP": mb.clinic.q_gap,
        "Q_REGULARITY": mb.clinic.q_regularity,
        "Q_BALANCE": mb.clinic.q_balance,
    }

    for name, metric in fair_metrics.items():
        assert metric(exp).value == pytest.approx(
            diagnostic.fair.metrics[name].value
        )

    for name, score in quality_scores.items():
        assert score(exp).value == pytest.approx(
            diagnostic.quality.scores[name].value
        )
