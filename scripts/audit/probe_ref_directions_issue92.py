# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only reference-direction fingerprint, issue #92.

No modification to pymoo, to MoeaBench production APIs, or to test baselines.
"""
from __future__ import annotations

import hashlib
import json
import platform

import numpy as np
import scipy
from pymoo.util.ref_dirs import get_reference_directions


def fingerprint(a):
    arr = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main():
    directions = np.asarray(
        get_reference_directions("energy", 3, 24, seed=3748133013), dtype=np.float64
    )
    assert directions.shape == (24, 3) and np.isfinite(directions).all()
    order = np.lexsort(directions.T[::-1])
    sorted_directions = directions[order]
    print(json.dumps({
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "directions_sorted_sha256": fingerprint(sorted_directions),
        "directions_original_sha256": fingerprint(directions),
        "directions_sorted_rounded13_sha256": fingerprint(np.round(sorted_directions, 13)),
        "directions_sorted_values": sorted_directions.tolist(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
