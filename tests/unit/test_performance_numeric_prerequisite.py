# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New fail-closed checks for current-revision numerical prerequisite.

Historical regression tests and numerical fixture files are not modified.
"""
import json
import os

import pytest

from moeabench import certification as cert


@pytest.fixture
def numeric_slot(tmp_path, monkeypatch):
    import moeabench.environment_verification as ev
    import moeabench.system as system

    monkeypatch.setattr(system, "version", lambda show=False: "1.0.0")
    monkeypatch.setattr(cert, "_revision", lambda: "candidate-source-sha")
    monkeypatch.setattr(ev, "snapshot", lambda: {"moeabench": "1.0.0", "numpy": "candidate"})
    for name in (
        "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE",
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
    ):
        monkeypatch.setenv(name, "test-profile-" + name)
    hashes = {
        relative: cert._hash_file(
            cert.Path(cert.__file__).resolve().parents[1] / relative
        )
        for relative in cert._NUMERICAL_FILES
    }
    record = {
        "schema": "moeabench-numeric-certificate",
        "version": "1.0.0",
        "source_revision": "candidate-source-sha",
        "counts": {"suite": "basic", "status": "PASSED"},
        "environment": {"moeabench": "1.0.0", "numpy": "candidate"},
        "numerical_profile": {
            name: os.environ.get(name) for name in (
                "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE",
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
            )
        },
        "numerical_fixtures_sha256": hashes,
    }
    path = tmp_path / "1.0.0" / "numeric-certificate.json"
    path.parent.mkdir()
    return tmp_path, path, record


def test_performance_requires_existing_passed_numeric_certificate(numeric_slot):
    root, path, record = numeric_slot
    with pytest.raises(RuntimeError, match="Missing numerical certificate"):
        cert.require_current_numeric_certificate(root)
    path.write_text(json.dumps(record))
    assert cert.require_current_numeric_certificate(root) == record
    record["counts"]["status"] = "FAILED"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="passed basic suite"):
        cert.require_current_numeric_certificate(root)


@pytest.mark.parametrize("field,value,message", [
    ("source_revision", "another-revision", "current source revision"),
    ("version", "0.18.0", "current source revision"),
    ("environment", {"moeabench": "wrong"}, "environment"),
    ("numerical_profile", {"wrong": "profile"}, "runtime profile"),
    ("numerical_fixtures_sha256", {}, "fixture hashes"),
])
def test_performance_rejects_mismatching_numeric_provenance(numeric_slot, field, value, message):
    root, path, record = numeric_slot
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match=message):
        cert.require_current_numeric_certificate(root)


def test_performance_blocks_before_measurement_when_numeric_gate_missing(
        numeric_slot, monkeypatch):
    root, _, _ = numeric_slot
    monkeypatch.setattr(cert, "_RESOURCE_ROOT", root)
    monkeypatch.setattr(cert, "load_performance_reference", lambda path: {
        "source_version": "0.18.0"
    })
    monkeypatch.setattr(cert, "_reference_runtime", lambda: pytest.fail(
        "Numerical prerequisite should prevent running the performance experiment"
    ))
    with pytest.raises(RuntimeError, match="Missing numerical certificate"):
        cert.certify_performance("historical.json", root / "result.json")
