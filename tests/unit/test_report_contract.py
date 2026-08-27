# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from moeabench.diagnostics.auditor import DiagnosticResult, FairAuditResult, QualityAuditResult
from moeabench.diagnostics.enums import DiagnosticStatus
from moeabench.diagnostics.fair import FairResult
from moeabench.diagnostics.qscore import QResult
from moeabench.core.base import Reportable


class ZMQInteractiveShell:
    pass


class TerminalInteractiveShell:
    pass


ColabShell = type("Shell", (), {"__module__": "google.colab._shell"})


class DummyReport(Reportable):
    def report(self, show: bool = True, **kwargs):
        content = "### Example\n\n" + self._report_block("Name  : value\nBold  : **literal**  ")
        return self._render_report(content, show=show, **kwargs)


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
    shown = []

    monkeypatch.setattr(Reportable, "_is_notebook", staticmethod(lambda: True))

    import IPython.display as ipd

    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj))
    monkeypatch.setattr(ipd, "Markdown", lambda text: text)

    res = DummyReport().report()

    assert isinstance(res, str)
    assert str(res).startswith("### Example")
    assert repr(res) == ""
    assert shown
    assert not shown[0].startswith("=")


def test_is_notebook_detects_jupyter(monkeypatch):
    import IPython

    monkeypatch.setattr(IPython, "get_ipython", lambda: ZMQInteractiveShell())

    assert Reportable._is_notebook() is True


def test_is_notebook_rejects_terminal_ipython(monkeypatch):
    import IPython

    monkeypatch.setattr(IPython, "get_ipython", lambda: TerminalInteractiveShell())

    assert Reportable._is_notebook() is False


def test_is_notebook_detects_colab_without_importing_it(monkeypatch):
    import IPython

    monkeypatch.setattr(IPython, "get_ipython", lambda: ColabShell())

    assert Reportable._is_notebook() is True


def test_is_notebook_rejects_regular_terminal(monkeypatch):
    import IPython

    monkeypatch.setattr(IPython, "get_ipython", lambda: None)

    assert Reportable._is_notebook() is False


def test_report_block_preserves_internal_alignment_and_rstrips_end():
    block = DummyReport()._report_block("A  : 1  \nB  : 2  ")

    assert block == "```text\nA  : 1  \nB  : 2\n```"


def test_plain_sober_handles_only_canonical_markdown_subset():
    content = (
        "# One\n## Two\n### Three\n#### Four\n##### Five\n###### Six\n"
        "#hashtag\n#######\n**outside**\n--- Legacy ---\n"
        "```text\n  aligned  : **literal**  \n```"
    )

    assert Reportable._to_plain_sober(content) == (
        "One\nTwo\nThree\nFour\nFive\nSix\n"
        "#hashtag\n#######\noutside\nLegacy\n"
        "  aligned  : **literal**  "
    )


def test_show_false_markdown_contract(monkeypatch):
    report = DummyReport()

    markdown = report.report(show=False, markdown=True)
    plain = report.report(show=False, markdown=False)

    assert "### Example" in markdown
    assert "```text" in markdown
    assert "###" not in plain
    assert "```" not in plain
    assert "Name  : value" in plain
    assert "Bold  : **literal**" in plain

    monkeypatch.setattr(Reportable, "_is_notebook", staticmethod(lambda: True))
    assert report.report(show=False) == markdown
    monkeypatch.setattr(Reportable, "_is_notebook", staticmethod(lambda: False))
    assert report.report(show=False) == plain


def test_rich_representations_use_centralized_formats():
    report = DummyReport()

    class PrettyPrinter:
        def __init__(self):
            self.value = None

        def text(self, value):
            self.value = value

    printer = PrettyPrinter()
    report._repr_pretty_(printer, cycle=False)

    assert "```" not in printer.value
    assert "```text" in report._repr_markdown_()


def test_terminal_display_is_plain_even_with_markdown_return_requested(
    monkeypatch, capsys
):
    monkeypatch.setattr(Reportable, "_is_notebook", staticmethod(lambda: False))

    returned = DummyReport().report(show=True, markdown=True)
    displayed = capsys.readouterr().out

    assert "```text" in returned
    assert "```" not in displayed
    assert "###" not in displayed
    assert "==================================" in displayed
    assert "Name  : value" in displayed


def test_notebook_honors_explicit_plain_representation(monkeypatch):
    import IPython.display as ipd

    shown = []
    monkeypatch.setattr(Reportable, "_is_notebook", staticmethod(lambda: True))
    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj))
    monkeypatch.setattr(ipd, "Markdown", lambda text: text)

    returned = DummyReport().report(show=True, markdown=False)

    assert "```" not in returned
    assert shown == [str(returned)]
