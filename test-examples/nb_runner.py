#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
nb_runner.py: Headless execution utility for Jupyter Notebooks.
"""

import sys
import os

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    print("Error: 'nbformat' and 'nbclient' are required to run notebooks.")
    print("Please install them using: pip install nbformat nbclient")
    sys.exit(1)

def run_notebook(nb_path):
    """Executes a notebook and returns True if successful, raises exception otherwise."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    
    try:
        client.execute()
        return True
    except Exception as e:
        print(f"\n[FAIL] Error executing notebook '{nb_path}':")
        print(f"Details: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 nb_runner.py <notebook_file.ipynb>")
        sys.exit(1)
    
    success = run_notebook(sys.argv[1])
    sys.exit(0 if success else 1)
