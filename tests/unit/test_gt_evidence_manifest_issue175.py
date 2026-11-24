# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New integrity tests for provisional GT evidence; no regression edits."""
import json

import pytest

from scripts.audit import gt_evidence_manifest as evidence


@pytest.fixture
def specimen(tmp_path, monkeypatch):
    artifact = tmp_path / "ground_truth.npz"
    artifact.write_bytes(b"independent numerical result bytes\x00\x01")
    monkeypatch.setattr(evidence, "revision", lambda root: "a" * 40)
    monkeypatch.setattr(evidence, "execution_environment", lambda: {
        "python": "test",
        "sha256": evidence.sha256_bytes(evidence.canonical_json({"python": "test"})),
    })
    config = {
        "problem": "DTLZ5", "parameters": {"M": 5, "N": 14},
        "seed": 17, "budget": 2000, "oracle": "independent_v1",
        "tolerance": 1e-10,
    }
    return artifact, config, tmp_path


def test_clean_capture_and_integrity_never_certify(specimen):
    artifact, config, root = specimen
    data = evidence.create_manifest(artifact, config, root)
    result = evidence.verify_manifest(
        artifact, data, expected_commit="a" * 40,
        expected_artifact_sha256=evidence.sha256_file(artifact))
    assert result == {
        "integrity": "pass", "errors": [],
        "scientific_certification": "unverified",
    }
    assert all(v == "unverified" for v in data["gates"].values())
    assert data["config_sha256"] == evidence.sha256_bytes(
        evidence.canonical_json(config))


def test_artifact_tampering_is_detected(specimen):
    artifact, config, root = specimen
    data = evidence.create_manifest(artifact, config, root)
    artifact.write_bytes(b"tampered\x00")
    result = evidence.verify_manifest(artifact, data)
    assert result["integrity"] == "fail"
    assert "artifact digest mismatch" in result["errors"]
    assert result["scientific_certification"] == "unverified"


def test_manifest_tampering_is_detected(specimen):
    artifact, config, root = specimen
    data = evidence.create_manifest(artifact, config, root)
    data["config"]["budget"] = 1000
    result = evidence.verify_manifest(artifact, data)
    assert result["integrity"] == "fail"
    assert "manifest digest mismatch" in result["errors"]
    assert "configuration digest mismatch" in result["errors"]


def test_identity_mismatch_and_fake_certification_are_rejected(specimen):
    artifact, config, root = specimen
    data = evidence.create_manifest(artifact, config, root)
    result = evidence.verify_manifest(
        artifact, data, expected_commit="b" * 40,
        expected_artifact_sha256="0" * 64)
    assert result["integrity"] == "fail"
    assert "source commit mismatch" in result["errors"]
    assert "expected artifact digest mismatch" in result["errors"]
    data["gates"]["mathematical_oracle"] = "pass"
    data["manifest_sha256"] = evidence.sha256_bytes(
        evidence.canonical_json({
            k: v for k, v in data.items() if k != "manifest_sha256"
        }))
    result = evidence.verify_manifest(artifact, data)
    assert "invalid gate state: this tool cannot certify gates" in result["errors"]


def test_required_experiment_provenance_cannot_be_omitted(specimen):
    artifact, config, root = specimen
    del config["seed"]
    with pytest.raises(ValueError, match="seed"):
        evidence.create_manifest(artifact, config, root)


def test_cli_capture_verify_and_corruption(specimen, capsys):
    artifact, config, root = specimen
    source = root / "config.json"
    output = root / "manifest.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    assert evidence.main([
        "capture", "--artifact", str(artifact), "--config", str(source),
        "--manifest", str(output), "--repo-root", str(root)]) == 0
    created = json.loads(output.read_text(encoding="utf-8"))
    assert created["source_commit"] == "a" * 40
    assert evidence.main([
        "verify", "--artifact", str(artifact), "--manifest", str(output),
        "--expected-commit", "a" * 40]) == 0
    assert "unverified" in capsys.readouterr().out
    artifact.write_bytes(b"different")
    assert evidence.main([
        "verify", "--artifact", str(artifact), "--manifest", str(output)]) == 1


def test_dirty_git_worktree_rejected(tmp_path, monkeypatch):
    def fake_check_output(argv, **kwargs):
        if "rev-parse" in argv:
            return "c" * 40 + "\n"
        return " M moeabench/mops/a.py\n"
    monkeypatch.setattr(evidence.subprocess, "check_output", fake_check_output)
    with pytest.raises(ValueError, match="uncommitted"):
        evidence.revision(tmp_path)


def test_capture_refuses_to_overwrite_existing_manifest(specimen):
    artifact, config, root = specimen
    source = root / "configuration.json"
    manifest = root / "record.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    manifest.write_text('{"protected": true}', encoding="utf-8")
    with pytest.raises(SystemExit):
        evidence.main([
            "capture", "--artifact", str(artifact), "--config", str(source),
            "--manifest", str(manifest), "--repo-root", str(root)])
    assert manifest.read_text(encoding="utf-8") == '{"protected": true}'


def test_capture_refuses_to_overwrite_configuration(specimen):
    artifact, config, root = specimen
    source = root / "configuration.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        evidence.main([
            "capture", "--artifact", str(artifact), "--config", str(source),
            "--manifest", str(source), "--repo-root", str(root)])
    assert source.read_bytes() == before
