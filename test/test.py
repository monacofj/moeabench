#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
test.py: Unified test orchestrator for MoeaBench.
Runs unit tests and/or example-based integration tests.
"""

import argparse
import subprocess
import os
import sys
import glob

def run_unit_tests():
    """Runs granular unit tests using pytest."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("Error: 'pytest' is required for unit tests.")
        print("Please install it using: pip install pytest")
        return False

    result = subprocess.run([sys.executable, "-m", "pytest", "test/unit"], check=False)
    return result.returncode == 0

def run_script_tests():
    """Runs all scripts in the examples/ directory."""
    print("\n" + "="*60)
    print("RUNNING SCRIPT TESTS")
    print("="*60)
    
    examples_dir = "examples"
    scripts = sorted(glob.glob(os.path.join(examples_dir, "*.py")))
    scripts = [s for s in scripts if os.path.basename(s) != "mb_path.py"]
    
    results = []
    for script in scripts:
        print(f"Testing script: {os.path.basename(script)}...", end=" ", flush=True)
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        res = subprocess.run([sys.executable, script], capture_output=True, text=True, env=env)
        
        if res.returncode == 0:
            print("[PASS]")
            results.append((script, True))
        else:
            print("[FAIL]")
            print(f"--- Stdout ---\n{res.stdout}")
            print(f"--- Stderr ---\n{res.stderr}")
            results.append((script, False))

    print_summary("SCRIPTS", results)
    return all(ok for _, ok in results)

def ensure_notebook_kernel():
    """Ensures that the 'python3' kernel is available for notebook execution."""
    try:
        import jupyter_client
        km = jupyter_client.kernelspec.KernelSpecManager()
        if "python3" in km.get_all_specs():
            return True
    except ImportError:
        pass

    print("Notebook kernel 'python3' not found. Attempting to register...")
    try:
        # Check if ipykernel is installed
        import ipykernel
        res = subprocess.run(
            [sys.executable, "-m", "ipykernel", "install", "--user", "--name", "python3"],
            capture_output=True, text=True
        )
        if res.returncode == 0:
            print("Successfully registered 'python3' kernel.")
            return True
        else:
            print(f"Failed to register kernel: {res.stderr}")
    except ImportError:
        print("Error: 'ipykernel' is required to register the notebook kernel.")
        print("Please install it using: pip install ipykernel")
    
    return False

def run_notebook_tests():
    """Runs all notebooks in the examples/ directory."""
    print("\n" + "="*60)
    print("RUNNING NOTEBOOK TESTS")
    print("="*60)
    
    # Check for core notebook execution dependencies
    try:
        import nbformat
        import nbclient
    except ImportError:
        print("Error: 'nbformat' and 'nbclient' are required to run notebook tests.")
        print("Please install them using: pip install nbformat nbclient")
        return False
    
    if not ensure_notebook_kernel():
        print("[ABORT] Could not ensure 'python3' kernel for notebooks.")
        return False

    examples_dir = "examples"
    test_dir = "test"
    notebooks = sorted(glob.glob(os.path.join(examples_dir, "*.ipynb")))
    
    nb_runner = os.path.join(test_dir, "nb_runner.py")
    results = []
    for nb in notebooks:
        print(f"Testing notebook: {os.path.basename(nb)}...", end=" ", flush=True)
        res = subprocess.run([sys.executable, nb_runner, nb], capture_output=True, text=True)
        
        if res.returncode == 0:
            print("[PASS]")
            results.append((nb, True))
        else:
            print("[FAIL]")
            print(f"--- Stdout ---\n{res.stdout}")
            print(f"--- Stderr ---\n{res.stderr}")
            results.append((nb, False))

    print_summary("NOTEBOOKS", results)
    return all(ok for _, ok in results)

def print_summary(label, results):
    print("\n" + "-"*60)
    print(f"SUMMARY OF {label}")
    print("-"*60)
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    for path, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"{status:<10} {os.path.basename(path)}")
    print("-"*40)
    print(f"Passed: {passed}, Failed: {failed}")

def main():
    # Path Resolution: Ensure we are running from the project root
    # regardless of where the script was called from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    os.chdir(project_root)

    parser = argparse.ArgumentParser(description="MoeaBench Test Orchestrator")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--scripts", action="store_true", help="Run example python scripts")
    parser.add_argument("--notebooks", action="store_true", help="Run example jupyter notebooks")
    parser.add_argument("--all", action="store_true", help="Run all tests (unit + scripts + notebooks)")

    args = parser.parse_args()

    # Default to --unit if no arguments provided
    if not (args.unit or args.scripts or args.notebooks or args.all):
        args.unit = True

    success = True

    if args.unit or args.all:
        if not run_unit_tests():
            success = False

    if args.scripts or args.all:
        if not run_script_tests():
            success = False

    if args.notebooks or args.all:
        if not run_notebook_tests():
            success = False

    if not success:
        print("\n[CRITICAL] Some tests failed!")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
