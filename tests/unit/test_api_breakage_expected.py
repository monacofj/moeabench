# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import moeabench as mb
from moeabench.diagnostics.auditor import DiagnosticResult, FairAuditResult, QualityAuditResult
from moeabench.diagnostics.enums import DiagnosticStatus
from moeabench.diagnostics.fair import FairResult
from moeabench.diagnostics.qscore import QResult


def _build_quality():
    return QualityAuditResult(
        scores={"Q_CLOSENESS": QResult(0.8, "Q_CLOSENESS", "ok")},
        mop_name="DTLZ2",
        k=50,
    )


def _build_diagnostic():
    q = _build_quality()
    f = FairAuditResult(metrics={"CLOSENESS": FairResult(0.1, "CLOSENESS", "ok")})
    return DiagnosticResult(
        q_audit_res=q,
        fair_audit_res=f,
        status=DiagnosticStatus.IDEAL_FRONT,
        description="ok",
        reproducibility=None,
        diagnostic_context=None,
    )


def test_removed_summary_methods():
    q = _build_quality()
    d = _build_diagnostic()
    assert not hasattr(q, "summary")
    assert not hasattr(d, "summary")

    with pytest.raises(AttributeError):
        getattr(q, "summary")
    with pytest.raises(AttributeError):
        getattr(d, "summary")


def test_removed_clinic_old_entrypoints():
    assert hasattr(mb.clinic, "audit")
    assert not hasattr(mb.clinic, "fair_audit")
    assert not hasattr(mb.clinic, "q_audit")


def test_removed_tier_public_stats_api():
    assert hasattr(mb.stats, "ranks")
    assert hasattr(mb.stats, "caste")
    assert hasattr(mb.stats, "tiers")
    assert not hasattr(mb.stats, "strata")
    assert not hasattr(mb.stats, "tier")
