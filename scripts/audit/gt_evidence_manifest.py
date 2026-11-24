# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Record and check immutable GT experiment evidence (ADR 0059).

This tool binds a *provisional* GT computation to source, inputs,
execution environment and an output digest. Hash verification is
NOT numerical stability certification, independent oracle validation,
or cross-platform reproducibility. Those are separate gates.

Capture:
  python scripts/audit/gt_evidence_manifest.py capture \\
    --artifact results.npz --config config.json --manifest results.manifest.json

Verify:
  python scripts/audit/gt_evidence_manifest.py verify \\
    --artifact results.npz --manifest results.manifest.json \\
    --expected-commit FULL_SHA
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


DEPENDENCIES = ("numpy", "scipy", "pymoo", "moocore", "moeabench")
SCHEMA = "moeabench_gt_evidence_v1"


def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision(repo_root):
    """Require a commit-pinned clean checkout to capture source-bound data."""
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            text=True, stderr=subprocess.DEVNULL
        ).strip()

    sha = git("rev-parse", "--verify", "HEAD")
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise ValueError("Expected a full 40-character Git commit SHA")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked source has uncommitted changes")
    return sha


def execution_environment():
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    env = {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": versions,
    }
    # NumPy reports its actual BLAS/LAPACK backend where available.
    try:
        import numpy as np
        from io import StringIO
        from contextlib import redirect_stdout
        output = StringIO()
        with redirect_stdout(output):
            np.show_config()
        env["numpy_build_config"] = output.getvalue()
    except (ImportError, AttributeError):
        env["numpy_build_config"] = None
    env["sha256"] = sha256_bytes(canonical_json(env))
    return env


def create_manifest(artifact, config, repo_root):
    """Capture identity/provenance without awarding scientific approval."""
    if not isinstance(config, dict) or not config:
        raise ValueError("Experiment config must be a nonempty JSON object")
    required = {"problem", "parameters", "seed", "budget", "oracle",
                "tolerance"}
    if not required.issubset(config):
        raise ValueError("Missing required experiment config keys: " +
                         ", ".join(sorted(required - set(config))))
    sha = revision(repo_root)
    env = execution_environment()
    data = {
        "schema": SCHEMA,
        "source_commit": sha,
        "artifact_name": Path(artifact).name,
        "artifact_sha256": sha256_file(artifact),
        "config": config,
        "config_sha256": sha256_bytes(canonical_json(config)),
        "environment": env,
        "gates": {
            "mathematical_oracle": "unverified",
            "execution_environment": "unverified",
            "result_reproducibility": "unverified",
        },
        "statement": (
            "Hash and provenance only. A matching manifest is not "
            "an independent scientific or numerical certificate."
        ),
    }
    data["manifest_sha256"] = sha256_bytes(canonical_json(data))
    return data


def verify_manifest(artifact, manifest, *, expected_commit=None,
                    expected_artifact_sha256=None):
    """Check recorded integrity only; return a structured outcome."""
    errors = []
    if manifest.get("schema") != SCHEMA:
        errors.append("unsupported schema")
    checksum = manifest.get("manifest_sha256")
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if checksum != sha256_bytes(canonical_json(payload)):
        errors.append("manifest digest mismatch")
    if manifest.get("artifact_name") != Path(artifact).name:
        errors.append("artifact name mismatch")
    if manifest.get("artifact_sha256") != sha256_file(artifact):
        errors.append("artifact digest mismatch")
    config = manifest.get("config")
    if not isinstance(config, dict) or manifest.get("config_sha256") != (
        sha256_bytes(canonical_json(config)) if isinstance(config, dict)
        else None
    ):
        errors.append("configuration digest mismatch")
    env = manifest.get("environment")
    if not isinstance(env, dict) or env.get("sha256") != (
        sha256_bytes(canonical_json({k: v for k, v in env.items()
                                     if k != "sha256"}))
        if isinstance(env, dict) else None
    ):
        errors.append("environment record digest mismatch")
    if expected_commit is not None and manifest.get(
        "source_commit") != expected_commit:
        errors.append("source commit mismatch")
    if expected_artifact_sha256 is not None and manifest.get(
        "artifact_sha256") != expected_artifact_sha256:
        errors.append("expected artifact digest mismatch")
    if manifest.get("gates") != {
        "mathematical_oracle": "unverified",
        "execution_environment": "unverified",
        "result_reproducibility": "unverified",
    }:
        errors.append("invalid gate state: this tool cannot certify gates")
    return {"integrity": "pass" if not errors else "fail",
            "errors": errors, "scientific_certification": "unverified"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--artifact", type=Path, required=True)
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    check = commands.add_parser("verify")
    check.add_argument("--artifact", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--expected-commit")
    check.add_argument("--expected-artifact-sha256")
    args = parser.parse_args(argv)

    if args.command == "capture":
        if args.manifest.resolve() in (
            args.artifact.resolve(), args.config.resolve()
        ):
            parser.error("Manifest must not overwrite artifact or configuration")
        if args.manifest.exists():
            parser.error("Refusing to replace an existing evidence manifest")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        data = create_manifest(args.artifact, config, args.repo_root)
        args.manifest.write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"integrity": "recorded",
                          "manifest_sha256": data["manifest_sha256"],
                          "scientific_certification": "unverified"}))
    else:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        outcome = verify_manifest(
            args.artifact, data, expected_commit=args.expected_commit,
            expected_artifact_sha256=args.expected_artifact_sha256,
        )
        print(json.dumps(outcome, sort_keys=True))
        if outcome["integrity"] != "pass":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
