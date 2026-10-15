"""Investigate environment-sensitive NSGA3 x DPF1 trajectories without altering baselines.

This script is intentionally diagnostic, not a pass/fail certification test:
it records which stage first departs across Python runtimes.
"""
import hashlib
import json
import platform
from importlib.metadata import version
from unittest.mock import patch

import moeabench.moeas._nsga_pymoo as nsga_backend

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions

import moeabench as mb
from moeabench.core.run import _calc_domination_mask
from moeabench.core.seeding import derive_component_seed


def array_record(values):
    arr = np.asarray(values, dtype="<f8")
    if arr.size == 0:
        arr = arr.reshape(0, 3)
    ordered = arr[np.lexsort(arr.T[::-1])] if arr.ndim == 2 else arr
    return {
        "shape": list(ordered.shape),
        "sha256": hashlib.sha256(ordered.tobytes()).hexdigest(),
        "values": ordered.tolist() if len(ordered) <= 30 else None,
    }


def main():
    seed = 12345
    mop = mb.mops.DPF1(M=3, D=2)
    ref_seed = derive_component_seed(seed, "nsga3.reference_directions")
    ref_dirs = get_reference_directions("energy", 3, 24, seed=ref_seed)
    print("ENV " + json.dumps({
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "pymoo": version("pymoo"),
        "moocore": version("moocore"),
        "ref_dirs_seed": ref_seed,
    }, sort_keys=True))
    print("REF_DIRS " + json.dumps(array_record(ref_dirs), sort_keys=True))
    exp = mb.experiment(mop=mop, moea=mb.moeas.NSGA3(
        population=24, generations=12, seed=seed,
    ))
    exp.run(repeat=1, silent=True)
    run = exp[0]
    for i in range(len(run)):
        all_f = np.asarray(run.pop(i).objectives)
        all_x = np.asarray(run.pop(i).variables)
        pymoo_nd = np.asarray(run.front(i))
        own_mask = ~_calc_domination_mask(all_f)
        own_nd = all_f[own_mask]
        print("GEN " + json.dumps({
            "index": i,
            "population_f": array_record(all_f),
            "population_x": array_record(all_x),
            "pymoo_nd": array_record(pymoo_nd),
            "recomputed_nd": array_record(own_nd),
            "component_seeds": run.component_seeds,
        }, sort_keys=True))
    print("FINAL " + json.dumps(array_record(exp.front()), sort_keys=True))

    # Diagnostic-only control: round away sub-ulp reference-direction
    # discrepancies before selection; production code remains unchanged.
    original = nsga_backend.get_reference_directions

    def rounded(*args, **kwargs):
        return np.round(original(*args, **kwargs), decimals=13)

    with patch.object(nsga_backend, "get_reference_directions", side_effect=rounded):
        controlled = mb.experiment(mop=mb.mops.DPF1(M=3, D=2),
                                   moea=mb.moeas.NSGA3(
                                       population=24, generations=12, seed=seed))
        controlled.run(repeat=1, silent=True)
    for i in range(len(controlled[0])):
        print("CONTROL " + json.dumps({
            "index": i,
            "population_f": array_record(controlled[0].pop(i).objectives)["sha256"],
            "population_x": array_record(controlled[0].pop(i).variables)["sha256"],
            "front": array_record(controlled[0].front(i))["sha256"],
            "front_count": len(controlled[0].front(i)),
        }, sort_keys=True))
    print("CONTROL_FINAL " + json.dumps(array_record(controlled.front()), sort_keys=True))


if __name__ == "__main__":
    main()
