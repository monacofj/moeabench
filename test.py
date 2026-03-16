#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unified test orchestrator for the MoeaBench scope/level test ontology."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import time


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"

SCOPES = ("unit", "integration", "stability")
LEVELS = ("smoke", "basic", "deep")


def _color(text: str, code: str) -> str:
    return f"{code}{text}{ANSI_RESET}"


def _parse_pytest_result(output: str):
    summary = {
        "collected": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "warnings": 0,
        "deselected": 0,
    }

    collected_match = re.search(r"collected\s+(\d+)\s+items", output)
    if collected_match:
        summary["collected"] = int(collected_match.group(1))

    for value, label in re.findall(r"(\d+)\s+(passed|failed|error|errors|skipped|warning|warnings|deselected)", output):
        count = int(value)
        key = "errors" if label in ("error", "errors") else ("warnings" if label in ("warning", "warnings") else label)
        summary[key] += count

    return summary


def _level_markers(level: str) -> list[str]:
    idx = LEVELS.index(level)
    return [f"level_{name}" for name in LEVELS[: idx + 1]]


def _scope_batches(scope: str | None) -> list[str]:
    if scope is None:
        return list(SCOPES)
    return [scope]


def _marker_expr(scope: str, level: str) -> str:
    if scope == "unit":
        return "scope_unit"
    levels = " or ".join(_level_markers(level))
    return f"scope_{scope} and ({levels})"


def _run_pytest(label: str, args: list[str], marker_expr: str):
    print("\n" + "=" * 60)
    print(_color(f"RUNNING {label} TESTS", ANSI_BOLD))
    print("=" * 60)

    env = {**os.environ, "PYTEST_ADDOPTS": (os.environ.get("PYTEST_ADDOPTS", "") + " --color=yes").strip()}
    pytest_args = [sys.executable, "-m", "pytest", "-m", marker_expr, "-v"]
    pytest_args.extend(args)
    started_at = time.perf_counter()
    proc = subprocess.Popen(
        pytest_args,
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
    elapsed = time.perf_counter() - started_at
    return proc.returncode == 0, parsed, elapsed


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, rem = divmod(minutes, 60)
    return f"{int(hours)}h {int(rem)}m {seconds % 60:.1f}s"


def _print_final_summary(rows):
    batch_width = 24
    status_width = 8
    time_width = 10
    print("\n" + "=" * 78)
    print(_color("FINAL SUMMARY", ANSI_BOLD))
    print("=" * 78)
    header = (
        f"{'BATCH':<{batch_width}} "
        f"{'STATUS':<{status_width}} "
        f"{'PASS/TOTAL':<12} "
        f"{'FAILED':>6} {'ERRORS':>6} {'SKIP':>6} {'WARN':>6} {'TIME':>{time_width}}"
    )
    print(_color(header, ANSI_BOLD))
    print("-" * 78)

    totals = {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "warnings": 0, "elapsed": 0.0}
    for row in rows:
        name = row["name"]
        status = row["status"]
        if status == "PASS":
            status_colored = _color(f"{status:<{status_width}}", ANSI_GREEN)
        elif status == "FAIL":
            status_colored = _color(f"{status:<{status_width}}", ANSI_RED)
        else:
            status_colored = _color(f"{status:<{status_width}}", ANSI_YELLOW)
        s = row["summary"]
        selected_total = s["passed"] + s["failed"] + s["errors"] + s["skipped"]
        pass_total = f"{s['passed']}/{selected_total}" if selected_total else "0/0"
        elapsed = _format_duration(row["elapsed"])
        print(
            f"{name:<{batch_width}} "
            f"{status_colored}"
            f"{pass_total:<12} "
            f"{s['failed']:>6} {s['errors']:>6} {s['skipped']:>6} {s['warnings']:>6} {elapsed:>{time_width}}"
        )
        totals["collected"] += s["collected"]
        totals["passed"] += s["passed"]
        totals["failed"] += s["failed"]
        totals["errors"] += s["errors"]
        totals["skipped"] += s["skipped"]
        totals["warnings"] += s["warnings"]
        totals["elapsed"] += row["elapsed"]

    overall = "PASS" if (totals["failed"] == 0 and totals["errors"] == 0 and totals["passed"] > 0) else "FAIL"
    overall_colored = _color(f"{overall:<{status_width}}", ANSI_GREEN if overall == "PASS" else ANSI_RED)
    print("-" * 78)
    selected_total = totals["passed"] + totals["failed"] + totals["errors"] + totals["skipped"]
    total_pass = f"{totals['passed']}/{selected_total}" if selected_total else "0/0"
    print(
        f"{'TOTAL':<{batch_width}} "
        f"{overall_colored}"
        f"{total_pass:<12} "
        f"{totals['failed']:>6} {totals['errors']:>6} {totals['skipped']:>6} {totals['warnings']:>6} {_format_duration(totals['elapsed']):>{time_width}}"
    )
    print("=" * 78)


class _CollectPlugin:
    def __init__(self):
        self.items = []

    def pytest_collection_finish(self, session):
        for item in session.items:
            marks = {m.name for m in item.iter_markers()}
            scopes = [scope for scope in SCOPES if f"scope_{scope}" in marks]
            levels = [level for level in LEVELS if f"level_{level}" in marks]
            self.items.append({
                "nodeid": item.nodeid,
                "scope": scopes[0] if scopes else None,
                "level": levels[0] if levels else None,
            })


def _collect_tests():
    try:
        import pytest
    except ImportError:
        print("Error: 'pytest' is required to list tests.")
        sys.exit(1)

    plugin = _CollectPlugin()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = pytest.main(["tests", "--collect-only", "-q"], plugins=[plugin])
    if code != 0:
        print(buffer.getvalue(), file=sys.stderr)
        raise SystemExit(code)
    return plugin.items


def _list_tests(scope: str | None, level: str | None):
    items = _collect_tests()
    filtered = []
    for item in items:
        if scope and item["scope"] != scope:
            continue
        if level and item["level"] != level:
            continue
        filtered.append(item)

    if not filtered:
        print("No tests matched the requested filters.")
        return

    grouped = {}
    for item in filtered:
        grouped.setdefault(item["scope"], []).append(item)

    for scope_name in SCOPES:
        group = grouped.get(scope_name, [])
        if not group:
            continue
        print(f"SCOPE: {scope_name}")
        for item in sorted(group, key=lambda row: (LEVELS.index(row["level"]), row["nodeid"])):
            print(f"- {item['nodeid']} [{item['level']}]")


def _selected_scope(args) -> str | None:
    for scope in SCOPES:
        if getattr(args, scope):
            return scope
    return None


def _selected_level(args) -> str | None:
    for level in LEVELS:
        if getattr(args, level):
            return level
    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="MoeaBench test orchestrator")
    parser.add_argument("--list", action="store_true", help="List collected tests by scope and level")
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument("--unit", action="store_true", help="Run or list the unit scope")
    scope_group.add_argument("--integration", action="store_true", help="Run or list through the integration scope")
    scope_group.add_argument("--stability", action="store_true", help="Run or list through the stability scope")
    level_group = parser.add_mutually_exclusive_group()
    level_group.add_argument("--smoke", action="store_true", help="Run or list the smoke level")
    level_group.add_argument("--basic", action="store_true", help="Run or list through the basic level")
    level_group.add_argument("--deep", action="store_true", help="Run or list through the deep level")

    args = parser.parse_args()
    scope = _selected_scope(args)
    level = _selected_level(args)

    if scope == "integration" and level == "deep":
        parser.error("`--integration` only supports `--smoke` or `--basic`.")

    if args.list:
        _list_tests(scope, level)
        return

    selected_scope = scope
    level = level or "smoke"

    rows = []
    success = True
    for batch_scope in _scope_batches(selected_scope):
        batch_level = level
        if batch_scope == "integration" and batch_level == "deep":
            batch_level = "basic"
        marker_expr = _marker_expr(batch_scope, batch_level)
        display_name = batch_scope.upper() if batch_scope == "unit" else f"{batch_scope.upper()} ({batch_level.upper()})"
        ok, summary, elapsed = _run_pytest(display_name, ["tests"], marker_expr)
        rows.append({
            "name": display_name,
            "status": "PASS" if ok else "FAIL",
            "summary": summary,
            "elapsed": elapsed,
        })
        if not ok:
            success = False

    _print_final_summary(rows)
    if not success:
        print("\n[CRITICAL] Some tests failed!")
        sys.exit(1)
    print("\nAll tasks completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
