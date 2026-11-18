# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Diagnostic-only NSGA3/DPF1 environment comparison, never a regression baseline.

Direction data were captured from the SUCCESSFUL numerical-basic CI run
44307380840 (2026-11-18) using Python 3.14.4, NumPy 2.5.4, SciPy 1.18.1,
pymoo 0.6.2 and seed=12345. These data MUST NOT be used as certified reference
outputs or inserted into the production NSGA3 wrapper. This is an isolated
diagnostic for differences between CI hosts and numerical-library kernels.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import platform
import sys
from importlib import metadata
from unittest.mock import patch

import numpy as np

import moeabench as mb
import moeabench.moeas._nsga_pymoo as backend

_EXPECTED_SHA = "d814a5d2313c9230e9ba1cfd1844ac6f90ced8e991ff99008e70191f3ace1d66"
_DIRECTIONS_BYTES = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAPA/AAAAAAAAAACa+HfoRgTGP9oB4kXufuo/AAAAAAAAAAB/3SYm/uLVP0GR7OyADuU/AAAAAAAAAACanIlXwBXgP83G7FB/1N8/AAAAAAAAAADKDXle/VjlP2zkDUMFTtU/AAAAAAAAAABolJJBJqXqP1+utflma8U/AAAAAAAAAAAAAAAAAADwPwAAAAAAAAAA+fxFEe2zxj/RJOil/K3GP413NJKFp+Q/woaoKda3xz8AAAAAAAAAAFDelXUKEuo/mklgLTJ0yD+n/+OdX5LdP43ba0uHM9Y/Uwj5BDoHyz/svcF+MT7pPwAAAAAAAAAA89rfh9U4yz9DazvYEdHjPwN4MhfjgsU/s3eWt0h40T+D899kxUbRP8mUiePxQN0/ER8zBnmG1j8AAAAAAAAAAHhw5nzDvOQ/gl4EFbjR2T9B0VGSkrXZP3mgU7Fq8cg/AXDbMSpy2j//RxLn6sbiPwAAAAAAAAAAPGyeUu+13T+It+cj/ZTIP/23bRuS/9U/BJiHWiRN4D8AAAAAAAAAAPnP8Eq3Zd8/xuIQR2AU4z9zOt5xP9fZPwAAAAAAAAAAp/zqFY/24z99phWCbcHKP+dmPiZWZMU/RHYu7JF75T8AAAAAAAAAAHcToyfcCNU/UdeHGt936T++ouCVgyDKPwAAAAAAAAAA7Dnao9Kq6j8AAAAAAAAAAE4Yl3C1VMU/AAAAAAAA8D8AAAAAAAAAAAAAAAAAAAAA"


def _frozen_directions():
    raw = base64.b64decode(_DIRECTIONS_BYTES, validate=True)
    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_SHA
    front = np.frombuffer(raw, dtype="<f8").reshape((24, 3)).copy()
    assert np.isfinite(front).all()
    return front


def _signature(a):
    values = np.asarray(a, dtype="<f8")
    if values.ndim == 2 and len(values):
        values = values[np.lexsort(values.T[::-1])]
    return {"shape": list(values.shape),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def _run_experiment(mode):
    used = []
    original_construct = backend.NSGA3
    original_directions = backend.get_reference_directions

    def inspect_construct(*args, **kwargs):
        used.append(np.asarray(kwargs["ref_dirs"], dtype=np.float64).copy())
        return original_construct(*args, **kwargs)

    def make_directions(*args, **kwargs):
        if mode == "frozen":
            assert args[:3] == ("energy", 3, 24)
            assert kwargs.get("seed") == 3748133013
            return _frozen_directions()
        return original_directions(*args, **kwargs)

    with patch.object(backend, "NSGA3", side_effect=inspect_construct), \
         patch.object(backend, "get_reference_directions", side_effect=make_directions):
        experiment = mb.experiment(
            mop=mb.mops.DPF1(M=3, D=2),
            moea=mb.moeas.NSGA3(population=24, generations=12, seed=12345),
        )
        experiment.run(repeat=1, silent=True)

    assert len(used) == 1, "Reference directions were not captured exactly once"
    if mode == "frozen":
        assert np.array_equal(used[0], _frozen_directions())
    run = experiment[0]
    return {
        "mode": mode,
        "directions": _signature(used[0]),
        "directions_sum": np.asarray(used[0]).sum(axis=1).tolist(),
        "final": _signature(experiment.front()),
        "generation_population": [_signature(run.pop(i).objectives)
                                  for i in range(len(run))],
    }


def main():
    runtime = io.StringIO()
    with contextlib.redirect_stdout(runtime):
        np.show_runtime()
    print("ENV " + json.dumps({
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": metadata.version("scipy"),
        "pymoo": metadata.version("pymoo"),
        "cpu_count": os.cpu_count(),
        "blas_environment": {k: os.getenv(k) for k in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
        )},
        "numpy_runtime": runtime.getvalue(),
    }, sort_keys=True))
    for mode in ("generated", "frozen"):
        print("RESULT " + json.dumps(_run_experiment(mode), sort_keys=True))


if __name__ == "__main__":
    main()
