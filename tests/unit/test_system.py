# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys

# Ensure the library is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from moeabench import mb

def test_version(capsys):
    """Verify return value and show behavior of version()."""
    v = mb.system.version(show=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert isinstance(v, str)
    assert len(v) > 0

    v2 = mb.system.version(show=True)
    captured = capsys.readouterr()
    assert v2 == v
    assert f"moeabench v{v}\n" == captured.out

def test_mb_object():
    """Verify the existence and structure of the 'mb' object."""
    assert hasattr(mb, 'system')
    assert hasattr(mb, 'mops')
    assert hasattr(mb, 'moeas')
    assert hasattr(mb, 'stats')
    assert hasattr(mb, 'view')

def test_dependency_check():
    """Verify that the dependency check function runs (does not crash)."""
    # This just checks it doesn't raise exception
    mb.system.check_dependencies()
