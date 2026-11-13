# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail on differences in core observations across pre/post FAIR cleanup."""
import argparse
import json
import math


def compare(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"Keys differ at {path}: {a.keys()} != {b.keys()}"
        for key in a:
            compare(a[key], b[key], path + "." + key)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"List length mismatch {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)), f"Type mismatch at {path}"
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (
            f"Numerical drift at {path}: baseline={a}, candidate={b}"
        )
    else:
        assert a == b, f"Different result at {path}: baseline={a}, candidate={b}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    with open(args.baseline, encoding="utf-8") as f:
        a = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        b = json.load(f)
    compare(a, b)
    print("FAIR-independent numeric compatibility probe: PASSED")
