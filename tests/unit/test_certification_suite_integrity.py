# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additive tests for the developer certification gate, issue #96.

Test runner behavior without executing pytest or touching the shipped manifest.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import moeabench.environment_verification as environment


@pytest.fixture
def runner():
    runner_path = Path(__file__).resolve().parents[2] / "test.py"
    spec = importlib.util.spec_from_file_location(
        "_moeabench_certification_test_runner", runner_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_certification_ignores_external_pytest_filters(monkeypatch, tmp_path, runner):
    manifest = tmp_path / "certified_environment.json"
    monkeypatch.setattr(environment, "_MANIFEST_PATH", manifest)
    monkeypatch.setattr(
        environment, "snapshot",
        lambda: {"python": "test-version", "moeabench": "test-version"},
    )
    filters = "-k only_one_test --ignore=tests/test_stability_basic.py"
    monkeypatch.setenv("PYTEST_ADDOPTS", filters)
    captured = {}

    def run_basic(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run_basic)
    assert runner._certify_environment() == 0
    assert captured["cmd"] == [sys.executable, runner.__file__, "--basic"]
    assert captured["env"]["PATH"]
    assert environment.snapshot() == json.loads(
        manifest.read_text(encoding="utf-8")
    )["versions"]
    # Filtering is removed only from the certification subprocess.
    assert runner.os.environ["PYTEST_ADDOPTS"] == filters


def test_failed_basic_does_not_overwrite_existing_manifest(
    monkeypatch, tmp_path, runner
):
    manifest = tmp_path / "certified_environment.json"
    original = '{"keep": "existing certified values"}\n'
    manifest.write_text(original, encoding="utf-8")
    monkeypatch.setattr(environment, "_MANIFEST_PATH", manifest)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k tiny")
    invoked = []

    def failed_basic(cmd, **kwargs):
        invoked.append(cmd)
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", failed_basic)
    assert runner._certify_environment() == 1
    assert invoked == [[sys.executable, runner.__file__, "--basic"]]
    assert manifest.read_text(encoding="utf-8") == original
    assert not manifest.with_suffix(".pending").exists()
