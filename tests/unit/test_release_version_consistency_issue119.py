# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Version-identity guard for the 1.0.0 certification candidate."""
from importlib.metadata import version as installed_version

import moeabench as mb
from moeabench.environment_verification import snapshot


def test_version_identity_matches_public_api_and_installed_metadata():
    expected = installed_version("moeabench")
    assert mb.system.version(show=False) == expected
    assert mb.__version__ == expected
    assert snapshot()["moeabench"] == expected
