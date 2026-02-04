#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
test.py: Unified test orchestrator for MoeaBench.
Runs functional unit tests and calibration tiers.
"""

import argparse
import subprocess
import os
import sys

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

    result = subprocess.run([sys.executable, "-m", "pytest", "tests/unit"], check=False)
    return result.returncode == 0

def run_tier_tests(tier_name):
    """Runs a specific calibration tier (light, smoke, heavy)."""
    tier_file = f"tests/test_{tier_name}_tier.py"
    if not os.path.exists(tier_file):
        print(f"Error: Tier file {tier_file} not found.")
        return False
        
    print("\n" + "="*60)
    print(f"RUNNING {tier_name.upper()} TIER TESTS")
    print("="*60)
    
    result = subprocess.run([sys.executable, "-m", "pytest", tier_file], check=False)
    return result.returncode == 0

def main():
    # Ensure we are running from the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="MoeaBench Test Orchestrator")
    parser.add_argument("--unit", action="store_true", help="Run functional unit tests")
    parser.add_argument("--light", action="store_true", help="Run Light Tier (Math Invariants)")
    parser.add_argument("--smoke", action="store_true", help="Run Smoke Tier (Regression)")
    parser.add_argument("--heavy", action="store_true", help="Run Heavy Tier (Statistical)")
    parser.add_argument("--all", action="store_true", help="Run all tests (excluding heavy)")

    args = parser.parse_args()

    # Default to --light if no specific tier is requested (foundational + math)
    if not (args.unit or args.light or args.smoke or args.heavy or args.all):
        args.light = True
    
    success = True

    # Unit Tests are MANDATORY for all paths as the system foundation
    if not run_unit_tests():
        success = False

    # Proceed with higher-level tiers only if foundation is solid
    if success:
        if args.light or args.all:
            if not run_tier_tests("light"):
                success = False

        if args.smoke or (args.all and not args.light): # Logical convenience
            if not run_tier_tests("smoke"):
                success = False

        if args.heavy:
            if not run_tier_tests("heavy"):
                success = False

    if not success:
        print("\n[CRITICAL] Some tests failed!")
        sys.exit(1)
    else:
        print("\nAll tasks completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
