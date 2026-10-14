# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Behavioral validation metadata for scientifically sensitive dependencies.

The versions recorded here are not package pins. They identify versions whose
relevant external-library behavior has been explicitly verified by MoeaBench's
dependency-verification battery.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _package_version
import warnings


VALIDATED_DEPENDENCIES = {
    "pymoo": "0.6.2",
}


class UnvalidatedDependencyWarning(UserWarning):
    """Warn that a sensitive dependency differs from its validated version."""


def dependency_validation_status(name: str) -> dict:
    """Return installed and behaviorally validated version metadata."""
    if name not in VALIDATED_DEPENDENCIES:
        raise KeyError(f"No dependency-validation record for {name!r}.")

    validated_version = VALIDATED_DEPENDENCIES[name]
    try:
        installed_version = _package_version(name)
    except PackageNotFoundError:
        installed_version = None

    return {
        "name": name,
        "installed_version": installed_version,
        "validated_version": validated_version,
        "is_validated": installed_version == validated_version,
    }


def dependency_validation_statuses() -> list[dict]:
    """Return validation metadata for all scientifically sensitive dependencies."""
    return [
        dependency_validation_status(name)
        for name in sorted(VALIDATED_DEPENDENCIES)
    ]


def warn_unvalidated_dependencies() -> None:
    """Warn when a sensitive dependency differs from its validated version.

    A warning does not block execution. Its purpose is to distinguish a version
    that satisfies packaging requirements from one whose relevant behavior has
    actually been checked against MoeaBench's scientific contracts.
    """
    for status in dependency_validation_statuses():
        installed = status["installed_version"]
        if installed is None or status["is_validated"]:
            continue
        warnings.warn(
            (
                f"{status['name']} {installed} differs from the version "
                f"behaviorally validated with MoeaBench "
                f"({status['validated_version']}). Scientific results may be "
                "affected by upstream behavioral changes. Run "
                "'python test.py --dependency-verify' before relying on "
                "scientific results."
            ),
            UnvalidatedDependencyWarning,
            stacklevel=2,
        )


__all__ = [
    "VALIDATED_DEPENDENCIES",
    "UnvalidatedDependencyWarning",
    "dependency_validation_status",
    "dependency_validation_statuses",
    "warn_unvalidated_dependencies",
]
