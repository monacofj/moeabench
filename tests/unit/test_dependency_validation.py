# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for dependency-version provenance and warnings."""

from __future__ import annotations

import pytest

import moeabench.dependency_validation as validation


def test_pymoo_validated_version_is_recorded():
    assert validation.VALIDATED_DEPENDENCIES["pymoo"] == "0.6.2"


def test_dependency_status_marks_matching_version_as_validated(monkeypatch):
    monkeypatch.setattr(validation, "_package_version", lambda name: "0.6.2")

    status = validation.dependency_validation_status("pymoo")

    assert status == {
        "name": "pymoo",
        "installed_version": "0.6.2",
        "validated_version": "0.6.2",
        "is_validated": True,
    }


def test_unvalidated_version_emits_actionable_warning(monkeypatch):
    monkeypatch.setattr(validation, "_package_version", lambda name: "0.6.3")

    with pytest.warns(
        validation.UnvalidatedDependencyWarning,
        match=r"python test\.py --dependency-verify",
    ):
        validation.warn_unvalidated_dependencies()


def test_unknown_dependency_has_no_validation_claim():
    with pytest.raises(KeyError, match="No dependency-validation record"):
        validation.dependency_validation_status("not-a-validated-dependency")
