# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from moeabench.diagnostics.auditor import DiagnosticResult, FairAuditResult, QualityAuditResult
from moeabench.diagnostics.enums import DiagnosticStatus
from moeabench.diagnostics.fair import FairResult
from moeabench.diagnostics.qscore import QResult
from moeabench.core.base import Reportable


def _make_quality_result() -> QualityAuditResult:
    scores = {
        "Q_CLOSENESS": QResult(0.8, "Q_CLOSENESS", "ok"),
        "Q_COVERAGE": QResult(0.7, "Q_COVERAGE", "ok"),
    }
    return QualityAuditResult(scores=scores, mop_name="DTLZ2", k=50)


def _make_fair_result() -> FairAuditResult:
    metrics = {
        "CLOSENESS": FairResult(0.1, "CLOSENESS", "ok"),
        "COVERAGE": FairResult(0.2, "COVERAGE", "ok"),
    }
    return FairAuditResult(metrics=metrics)


def test_quality_report_contract():
    res = _make_quality_result()
    assert hasattr(res, "report")
    assert not hasattr(res, "summary")
    brief = res.report(show=False, full=False)
    full = res.report(show=False, full=True)
    assert isinstance(brief, str) and brief.strip()
    assert isinstance(full, str) and full.strip()


def test_diagnostic_report_contract():
    q = _make_quality_result()
    f = _make_fair_result()
    res = DiagnosticResult(
        q_audit_res=q,
        fair_audit_res=f,
        status=DiagnosticStatus.IDEAL_FRONT,
        description="ok",
        reproducibility=None,
        diagnostic_context=None,
    )
    assert hasattr(res, "report")
    assert not hasattr(res, "summary")
    brief = res.report(show=False, full=False)
    full = res.report(show=False, full=True)
    assert isinstance(brief, str) and brief.strip()
    assert isinstance(full, str) and full.strip()


def test_render_report_returns_silent_string_in_notebook(monkeypatch):
    class DummyReport(Reportable):
        def report(self, show: bool = True, **kwargs):
            return self._render_report("### Example", show=show, **kwargs)

    shown = []

    monkeypatch.setattr(Reportable, "_is_notebook", staticmethod(lambda: True))

    import IPython.display as ipd

    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj))
    monkeypatch.setattr(ipd, "Markdown", lambda text: text)

    res = DummyReport().report()

    assert isinstance(res, str)
    assert str(res) == "### Example"
    assert repr(res) == ""
    assert shown
