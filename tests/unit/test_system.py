# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys

# Ensure the library is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from MoeaBench import mb

def test_version():
    """Verify that the library version is a non-empty string."""
    v = mb.system.version()
    assert isinstance(v, str)
    assert len(v) > 0
    print(f"Library Version: {v}")

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
