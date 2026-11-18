#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify the CI reference CPU/BLAS profile before numerical certification.

This is an environment preflight only: it does not alter NumPy, OpenBLAS,
pymoo, the solver, any fixtures or the certified dependency manifest.
See issue #92 and ADR 0050 for the scope of the reproducibility evidence.
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_info


def main():
    required = {
        "NPY_DISABLE_CPU_FEATURES": "X86_V4",
        "OPENBLAS_CORETYPE": "HASWELL",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for key, value in required.items():
        if os.environ.get(key) != value:
            raise RuntimeError(
                f"Uncertified numeric reference configuration: {key} must be {value!r}"
            )

    from numpy.lib.introspect import opt_func_info

    power = opt_func_info(func_name="power", signature="float64")
    current = power.get("power", {}).get("ddd", {}).get("current")
    if current != "baseline(X86_V2)":
        raise RuntimeError(
            f"Unexpected NumPy float64 power dispatch: {current!r}"
        )

    # Ensure that NumPy's and SciPy's BLAS libraries are actually loaded
    # and expose the requested kernel, not merely that an env var was set.
    a = np.eye(3)
    _ = a @ a
    _ = np.linalg.solve(a, np.ones(3))
    _ = scipy.linalg.solve(a, np.ones(3))
    libraries = [
        entry for entry in threadpool_info()
        if entry.get("internal_api") == "openblas"
    ]
    if not libraries:
        raise RuntimeError("No OpenBLAS runtime was detected")
    for item in libraries:
        if str(item.get("architecture", "")).lower() != "haswell":
            raise RuntimeError(f"Unexpected OpenBLAS CPU kernel: {item!r}")
        if item.get("num_threads") != 1:
            raise RuntimeError(f"OpenBLAS must be single-threaded: {item!r}")

    print(json.dumps({
        "numpy": np.__version__,
        "scipy": scipy.linalg.__version__ if hasattr(scipy.linalg, "__version__") else None,
        "power_dispatch": current,
        "openblas_kernels": [
            {
                "filepath": item.get("filepath"),
                "architecture": item.get("architecture"),
                "num_threads": item.get("num_threads"),
            }
            for item in libraries
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
