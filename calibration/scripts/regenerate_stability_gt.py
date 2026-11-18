# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate analytical GT geometry limits used by stability tests.

The stability contract compares the current deterministic analytical Pareto-front
sampling against the representative GT points frozen in the calibration audit.
The stored values are upper bounds, not exact fingerprints: this helper widens
only bounds that the current intentional behavior exceeds and leaves tighter
current observations inside their existing envelopes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import moeabench as mb


AUDIT_PATH = os.path.join(
    PROJ_ROOT, "tests", "calibration_reference_audit_v0.13.2.json"
)
REFERENCE_PATH = os.path.join(PROJ_ROOT, "tests", "stability_reference_gt.json")
LEVEL = "basic"


def _build_problem(mop_name):
    mop_cls = getattr(mb.mops, mop_name)
    try:
        return mop_cls(M=3)
    except TypeError:
        return mop_cls(M=3, D=2)


def _geometry_signature(problem_name, audit_data):
    """Return directed max nearest-neighbor distances to the frozen GT sample."""
    problem_data = audit_data["problems"][problem_name]
    expected = np.asarray(problem_data["gt_points"], dtype=float)
    mop = _build_problem(problem_name)
    observed = np.asarray(mop.pf(n_points=len(expected)), dtype=float)

    forward = cdist(observed, expected).min(axis=1)
    backward = cdist(expected, observed).min(axis=1)
    return {
        "forward_max": float(np.max(forward)),
        "backward_max": float(np.max(backward)),
    }


def regenerate(problems=None, write=False):
    """Recompute selected analytical GT stability limits."""
    with open(AUDIT_PATH, "r", encoding="utf-8") as handle:
        audit_data = json.load(handle)
    with open(REFERENCE_PATH, "r", encoding="utf-8") as handle:
        references = json.load(handle)

    available = references[LEVEL]
    selected = list(problems) if problems else list(available)

    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(
            "Unknown stability GT problem(s): " + ", ".join(unknown)
        )

    for problem_name in selected:
        old = available[problem_name]
        new = _geometry_signature(problem_name, audit_data)
        print(
            f"{problem_name}: "
            f"forward {old['forward_max']:.17g} -> {new['forward_max']:.17g}; "
            f"backward {old['backward_max']:.17g} -> {new['backward_max']:.17g}"
        )
        if write:
            # These references are stability envelopes, not exact signatures.
            # Preserve an existing bound when the current value is lower; only
            # widen a bound when intentional behavior now exceeds it.
            available[problem_name] = {
                key: max(old[key], new[key])
                for key in ("forward_max", "backward_max")
            }

    if write:
        with open(REFERENCE_PATH, "w", encoding="utf-8") as handle:
            json.dump(references, handle, indent=2)
            handle.write("\n")
        print(f"Updated {REFERENCE_PATH}")
    else:
        print("Dry run only; pass --write to update the reference file.")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate analytical GT geometry stability limits."
    )
    parser.add_argument(
        "--problem",
        dest="problems",
        action="append",
        help="Problem name to refresh; may be supplied more than once.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write regenerated limits back to tests/stability_reference_gt.json.",
    )
    args = parser.parse_args()
    regenerate(problems=args.problems, write=args.write)


if __name__ == "__main__":
    main()
