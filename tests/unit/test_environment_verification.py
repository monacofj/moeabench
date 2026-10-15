# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Environment verification regression tests."""
from types import SimpleNamespace
from moeabench import environment_verification as ev


def test_version_comparison(monkeypatch):
    monkeypatch.setattr(ev, "snapshot", lambda: {"python": "3", "numpy": "2"})
    monkeypatch.setattr(ev, "certified_manifest", lambda: {"schema": 1, "versions": {"python": "3", "numpy": "1"}})
    monkeypatch.setattr(ev, "_version_cache", None)
    assert ev.verify_version()["status"] == "MISMATCH"
    assert "numpy" in ev.verify_version()["differences"]


def test_experiment_provenance(monkeypatch):
    monkeypatch.setattr(ev, "snapshot", lambda: {"python": "3", "numpy": "2"})
    exp = SimpleNamespace()
    assert ev.verify_experiment(exp)["status"] == "UNKNOWN"
    exp._execution_environment = {"python": "3", "numpy": "1"}
    assert ev.verify_experiment(exp)["status"] == "MISMATCH"
    exp._execution_environment["numpy"] = "2"
    assert ev.verify_experiment(exp)["status"] == "MATCH"


def test_stability_cache_invalidation(monkeypatch):
    monkeypatch.setattr(ev, "snapshot", lambda: {"numpy": "1"})
    monkeypatch.setattr(ev, "_stability_cache", {"fingerprint": {"numpy": "1"}, "result": {"status": "PASSED"}})
    assert ev.stability_status() == "PASSED"
    monkeypatch.setattr(ev, "snapshot", lambda: {"numpy": "2"})
    assert ev.stability_status() == "UNCHECKED"


def test_export_load_preserves_execution_provenance(tmp_path):
    import moeabench as mb
    from moeabench.environment_verification import snapshot
    first = mb.experiment()
    first._execution_environment = snapshot()
    path = tmp_path / "provenance.zip"
    first.save(str(path), mode="config")
    second = mb.experiment()
    second.load(str(path))
    assert second._execution_environment == first._execution_environment
    assert second._environment_comparison["status"] == "MATCH"


def test_report_reads_state_without_rechecking(monkeypatch):
    import moeabench as mb
    exp = mb.experiment()
    def unexpected(**kwargs):
        raise AssertionError("report attempted version verification")
    monkeypatch.setattr(ev, "verify_version", unexpected)
    content = exp.report(show=False)
    assert "Reproducibility" in content
    assert "Numerical stability" in content
