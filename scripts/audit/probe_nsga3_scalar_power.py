# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Diagnostic-only full NSGA3 run with scalar power inside pymoo variation.

No production wrapper, regression test, reference fixture or threshold changes.
This evaluates whether the observed SBX np.power drift is causally sufficient
to produce the historical final-front divergence.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
from unittest.mock import patch

import numpy as np
import pymoo.operators.crossover.sbx as sbx_module
import pymoo.operators.mutation.pm as pm_module

import moeabench as mb

PASS_FRONT_SHA = "77dc14fcde9e939ad162ec4513254a5809a5dd6b76fd32e7a58e8c5b97205086"


def scalar_power(a, b):
    return np.frompyfunc(math.pow, 2, 1)(a, b).astype(float)


def signature(front):
    values = np.asarray(front, dtype="<f8")
    values = values[np.lexsort(values.T[::-1])]
    return {"shape": list(values.shape),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def main():
    original_cross_sbx = sbx_module.cross_sbx
    original_mut_pm = pm_module.mut_pm

    def scalar_cross_sbx(*args, **kwargs):
        with patch.object(np, "power", side_effect=scalar_power):
            return original_cross_sbx(*args, **kwargs)

    def scalar_mut_pm(*args, **kwargs):
        with patch.object(np, "power", side_effect=scalar_power):
            return original_mut_pm(*args, **kwargs)

    for mode in ("original", "sbx_scalar_power", "sbx_and_pm_scalar_power"):
        with ExitStack() as stack:
            if mode != "original":
                stack.enter_context(
                    patch.object(sbx_module, "cross_sbx", side_effect=scalar_cross_sbx))
            if mode == "sbx_and_pm_scalar_power":
                stack.enter_context(
                    patch.object(pm_module, "mut_pm", side_effect=scalar_mut_pm))
            exp = mb.experiment(
                mop=mb.mops.DPF1(M=3, D=2),
                moea=mb.moeas.NSGA3(population=24, generations=12, seed=12345),
            )
            exp.run(repeat=1, silent=True)

        final = signature(exp.front())
        generation_hashes = [
            signature(exp[0].pop(g).objectives)
            for g in range(len(exp[0]))
        ]
        print("RESULT " + json.dumps({
            "mode": mode,
            "final": final,
            "matches_historical_passing_front": final["sha256"] == PASS_FRONT_SHA,
            "generations": generation_hashes,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
