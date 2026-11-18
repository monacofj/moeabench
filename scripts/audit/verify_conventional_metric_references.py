#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independently verify issue #97 fixture by elementary stdlib mathematics.

This script intentionally imports neither MoeaBench, pymoo, NumPy nor SciPy.
It does not regenerate/overwrite the frozen fixture.
"""
from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "conventional_metric_reference_v1.json"


def distance(a, b, *, positive=False):
    terms = [(max(0.0, x - y) if positive else x - y) for x, y in zip(a, b)]
    return math.sqrt(sum(x * x for x in terms))


def nearest(a, collection, *, positive=False):
    return min(distance(a, b, positive=positive) for b in collection)


def average(values):
    values = list(values)
    return sum(values) / len(values)


def wasserstein_1d(xs, ys):
    """Exact empirical equal-weight 1D transportation via quantile intervals."""
    a, b = sorted(xs), sorted(ys)
    edges = sorted(set([0.0, 1.0]
                       + [i / len(a) for i in range(1, len(a))]
                       + [i / len(b) for i in range(1, len(b))]))
    total = 0.0
    for lo, hi in zip(edges, edges[1:]):
        midpoint = (lo + hi) / 2
        x = a[min(int(midpoint * len(a)), len(a) - 1)]
        y = b[min(int(midpoint * len(b)), len(b) - 1)]
        total += (hi - lo) * abs(x - y)
    return total


def exact_hv(front):
    """Inclusion-exclusion of axis-aligned dominated boxes with corner 1.1."""
    dim = len(front[0])
    total = 0.0
    for size in range(1, len(front) + 1):
        for subset in combinations(front, size):
            maxima = [max(point[i] for point in subset) for i in range(dim)]
            volume = math.prod(max(0.0, 1.1 - val) for val in maxima)
            total += volume if size % 2 else -volume
    return total


def reference_values(case):
    front, reference = case["front"], case["reference"]
    dim = case["objectives"]
    raw = exact_hv(front)
    return {
        "gd": average(nearest(p, reference) for p in front),
        "igd": average(nearest(r, front) for r in reference),
        "gdplus": average(nearest(p, reference, positive=True) for p in front),
        "igdplus": average(
            min(distance(p, r, positive=True) for p in front)
            for r in reference
        ),
        "emd": average(
            wasserstein_1d(
                [p[i] for p in front],
                [r[i] for r in reference],
            )
            for i in range(dim)
        ),
        "hv_raw": raw,
        "hv_rel": raw / exact_hv(reference),
        "hv_abs": raw / exact_hv(case["calibrated_gt"]),
    }


def main():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema"] == "moeabench_conventional_metric_reference_v1"
    max_error = 0.0
    count = 0
    for case in payload["cases"]:
        actual = reference_values(case)
        for name, expected in case["expected"].items():
            error = abs(actual[name] - expected)
            assert error < 1e-12, (case["id"], name, expected, actual[name])
            max_error = max(error, max_error)
            count += 1
    for case in payload["zero_cases"]:
        f, ref = case["front"], case["reference"]
        values = {
            "gd": average(nearest(p, ref) for p in f),
            "igd": average(nearest(r, f) for r in ref),
            "gdplus": average(nearest(p, ref, positive=True) for p in f),
            "igdplus": average(min(distance(p, r, positive=True) for p in f)
                               for r in ref),
            "emd": average(wasserstein_1d([p[i] for p in f],
                        [r[i] for r in ref]) for i in range(len(f[0]))),
        }
        for name, expected in case["expected"].items():
            assert values[name] == expected, (case["id"], name, values[name])
            count += 1
    print(f"Verified {count} frozen conventional-metric values independently; "
          f"maximum absolute arithmetic difference: {max_error:.3g}")


if __name__ == "__main__":
    main()
