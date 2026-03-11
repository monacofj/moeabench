# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Helper module to allow running examples without installing the package.
It adds the parent directory to sys.path so 'import moeabench' works.
"""
import sys
import os

# Add parent directory (project root) at highest priority in sys.path.
# This ensures examples import the local source tree, not an installed package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
