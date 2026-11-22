# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent advisory checks for sensitive dependency drift (issue #63).

Existing environment-verification regressions remain unchanged.
"""

import pytest
import warnings
from moeabench import environment_verification as ev


def test_pymoo_version_drift_prompts_manual_behavior_verification(monkeypatch):
    result = {
        "status": "MISMATCH",
        "current": {"pymoo": "0.6.3", "moeabench": "1.0.0"},
        "differences": {"pymoo": {"certified": "0.6.2", "current": "0.6.3"}},
    }
    monkeypatch.setattr(ev, "_version_cache", {"result": result})
    monkeypatch.setattr(ev, "_warned_fingerprints", set())
    monkeypatch.setattr(ev, "stability_status", lambda: "UNCHECKED")

    with pytest.warns(ev.EnvironmentWarning, match="--dependency-verify"):
        ev.warn_on_unverified()

    # The existing nonrepeating-advisory contract is preserved.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ev.warn_on_unverified()
    assert not record


def test_non_pymoo_drift_does_not_offer_unrelated_dependency_battery(monkeypatch):
    result = {
        "status": "MISMATCH",
        "current": {"pymoo": "0.6.2", "numpy": "2.6.0"},
        "differences": {"numpy": {"certified": "2.5.4", "current": "2.6.0"}},
    }
    monkeypatch.setattr(ev, "_version_cache", {"result": result})
    monkeypatch.setattr(ev, "_warned_fingerprints", set())
    monkeypatch.setattr(ev, "stability_status", lambda: "UNCHECKED")

    with pytest.warns(ev.EnvironmentWarning) as record:
        ev.warn_on_unverified()
    assert len(record) == 1
    assert "mb.system.verify_stability()" in str(record[0].message)
    assert "--dependency-verify" not in str(record[0].message)
