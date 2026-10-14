# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate deterministic solver-front baselines used by stability tests."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import moeabench as mb


REFERENCE_PATH = os.path.join(PROJ_ROOT, "tests", "stability_reference_fronts.json")


def _build_problem(mop_name):
    mop_cls = getattr(mb.mops, mop_name)
    try:
        return mop_cls(M=3)
    except TypeError:
        return mop_cls(M=3, D=2)


def _regenerate_front(entry):
    mop = _build_problem(entry["mop"])
    algorithm = getattr(mb.moeas, entry["algorithm"])(
        seed=entry["seed"],
        population=entry["population"],
        generations=entry["generations"],
    )
    experiment = mb.experiment(mop=mop)
    experiment.moea = algorithm
    experiment.run(repeat=1, silent=True)
    return np.asarray(experiment.front(), dtype=float).tolist()


def regenerate(algorithms=None, write=False):
    """Recompute selected algorithmic stability fronts."""
    with open(REFERENCE_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    selected = set(algorithms or [])
    changed = 0

    for level, entries in data.items():
        for entry in entries:
            if selected and entry["algorithm"] not in selected:
                continue

            label = (
                f'{level}: {entry["algorithm"]} x {entry["mop"]} '
                f'(seed={entry["seed"]}, pop={entry["population"]}, '
                f'gen={entry["generations"]})'
            )
            print(f"Regenerating {label}...")
            entry["front"] = _regenerate_front(entry)
            changed += 1

    if not changed:
        raise SystemExit("No stability entries matched the requested algorithms.")

    if write:
        with open(REFERENCE_PATH, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        print(f"Updated {REFERENCE_PATH}")
    else:
        print("Dry run only; pass --write to update the reference file.")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate deterministic solver fronts for stability tests."
    )
    parser.add_argument(
        "--algorithm",
        dest="algorithms",
        action="append",
        help="Algorithm name to refresh; may be supplied more than once.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write regenerated fronts back to tests/stability_reference_fronts.json.",
    )
    args = parser.parse_args()
    regenerate(algorithms=args.algorithms, write=args.write)


if __name__ == "__main__":
    main()
