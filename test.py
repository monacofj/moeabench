#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
test.py: Unified test orchestrator for moeabench.
Runs functional unit tests and calibration tiers.
"""

import argparse
import subprocess
import os
import sys
import re

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{ANSI_RESET}"


def _parse_pytest_result(output: str):
    """Extracts collected/tests summary counters from pytest output."""
    summary = {
        "collected": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
    }

    collected_match = re.search(r"collected\s+(\d+)\s+items", output)
    if collected_match:
        summary["collected"] = int(collected_match.group(1))

    for value, label in re.findall(r"(\d+)\s+(passed|failed|error|errors|skipped|warning|warnings)", output):
        count = int(value)
        key = "errors" if label in ("error", "errors") else ("warnings" if label in ("warning", "warnings") else label)
        summary[key] += count

    return summary


def _run_pytest(label: str, args: list[str]):
    """Runs pytest and returns (ok, parsed_summary)."""
    print("\n" + "=" * 60)
    print(_color(f"RUNNING {label} TESTS", ANSI_BOLD))
    print("=" * 60)

    env = {**os.environ, "PYTEST_ADDOPTS": (os.environ.get("PYTEST_ADDOPTS", "") + " --color=yes").strip()}
    proc = subprocess.Popen(
        [sys.executable, "-m", "pytest", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    captured = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        captured.append(line)

    proc.wait()
    output = "".join(captured)
    parsed = _parse_pytest_result(output)
    return proc.returncode == 0, parsed

def run_unit_tests():
    """Runs granular unit tests using pytest."""
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("Error: 'pytest' is required for unit tests.")
        print("Please install it using: pip install pytest")
        return False, {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "warnings": 0,
        }

    return _run_pytest("UNIT", ["tests/unit"])

def run_tier_tests(tier_name):
    """Runs a specific calibration tier (light, smoke, heavy)."""
    tier_file = f"tests/test_{tier_name}_tier.py"
    if not os.path.exists(tier_file):
        print(f"Error: Tier file {tier_file} not found.")
        return False, {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "warnings": 0,
        }

    return _run_pytest(f"{tier_name.upper()} TIER", [tier_file])


def _print_final_summary(rows):
    """Prints an aligned final summary table."""
    print("\n" + "=" * 78)
    print(_color("FINAL SUMMARY", ANSI_BOLD))
    print("=" * 78)
    print(_color(f"{'BATCH':<14} {'STATUS':<10} {'PASS/TOTAL':<12} {'FAILED':>6} {'ERRORS':>6} {'WARN':>6}", ANSI_BOLD))
    print("-" * 78)

    totals = {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "warnings": 0}
    for row in rows:
        name = row["name"]
        status = row["status"]
        if status == "PASS":
            status_colored = _color(status, ANSI_GREEN)
        elif status == "FAIL":
            status_colored = _color(status, ANSI_RED)
        else:
            status_colored = _color(status, ANSI_YELLOW)
        s = row["summary"]
        pass_total = f"{s['passed']}/{s['collected']}" if s["collected"] else "0/0"
        print(f"{name:<14} {status_colored:<19} {pass_total:<12} {s['failed']:>6} {s['errors']:>6} {s['warnings']:>6}")
        totals["collected"] += s["collected"]
        totals["passed"] += s["passed"]
        totals["failed"] += s["failed"]
        totals["errors"] += s["errors"]
        totals["warnings"] += s["warnings"]

    overall = "PASS" if (totals["failed"] == 0 and totals["errors"] == 0 and totals["passed"] > 0) else "FAIL"
    overall_colored = _color(overall, ANSI_GREEN if overall == "PASS" else ANSI_RED)
    print("-" * 78)
    total_pass = f"{totals['passed']}/{totals['collected']}"
    print(f"{'TOTAL':<14} {overall_colored:<19} {total_pass:<12} {totals['failed']:>6} {totals['errors']:>6} {totals['warnings']:>6}")
    print("=" * 78)

def main():
    # Ensure we are running from the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="moeabench Test Orchestrator")
    parser.add_argument("--unit", action="store_true", help="Run functional unit tests")
    parser.add_argument("--light", action="store_true", help="Run Light Tier (Math Invariants)")
    parser.add_argument("--smoke", action="store_true", help="Run Smoke Tier (Convergence Regression)")
    parser.add_argument("--regression", action="store_true", help="Run Regression Tier (Numerical Integrity)")
    parser.add_argument("--heavy", action="store_true", help="Run Heavy Tier (Statistical)")
    parser.add_argument("--all", action="store_true", help="Run all tests (excluding heavy)")

    args = parser.parse_args()

    # Default logic (Daily Validation): Unit + Light + Regression
    if not (args.unit or args.light or args.smoke or args.regression or args.heavy or args.all):
        args.light = True
        args.regression = True
    
    success = True
    summary_rows = []

    # Unit Tests are MANDATORY for all paths as the system foundation
    ok, unit_summary = run_unit_tests()
    summary_rows.append({"name": "UNIT", "status": "PASS" if ok else "FAIL", "summary": unit_summary})
    if not ok:
        success = False

    # Proceed with higher-level tiers only if foundation is solid
    if success:
        if args.light or args.all:
            ok, s = run_tier_tests("light")
            summary_rows.append({"name": "LIGHT", "status": "PASS" if ok else "FAIL", "summary": s})
            if not ok:
                success = False

        if args.regression or args.all:
            ok, s = run_tier_tests("regression")
            summary_rows.append({"name": "REGRESSION", "status": "PASS" if ok else "FAIL", "summary": s})
            if not ok:
                success = False

        if args.smoke or args.all:
            ok, s = run_tier_tests("smoke")
            summary_rows.append({"name": "SMOKE", "status": "PASS" if ok else "FAIL", "summary": s})
            if not ok:
                success = False

        if args.heavy:
            ok, s = run_tier_tests("heavy")
            summary_rows.append({"name": "HEAVY", "status": "PASS" if ok else "FAIL", "summary": s})
            if not ok:
                success = False
    else:
        if args.light or args.all:
            summary_rows.append({"name": "LIGHT", "status": "SKIPPED", "summary": {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "warnings": 0}})
        if args.regression or args.all:
            summary_rows.append({"name": "REGRESSION", "status": "SKIPPED", "summary": {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "warnings": 0}})
        if args.smoke or args.all:
            summary_rows.append({"name": "SMOKE", "status": "SKIPPED", "summary": {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "warnings": 0}})
        if args.heavy:
            summary_rows.append({"name": "HEAVY", "status": "SKIPPED", "summary": {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "warnings": 0}})

    _print_final_summary(summary_rows)

    if not success:
        print("\n[CRITICAL] Some tests failed!")
        sys.exit(1)
    else:
        print("\nAll tasks completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
