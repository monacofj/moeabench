# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only NSGA3/DPF1 variation signature probe.

Wrap the real pymoo SBX and PolynomialMutation operators, record inputs,
outputs and available RNG state, but return their unmodified results.
No baseline, production code, or regression test is changed.
"""
import hashlib
import json

import numpy as np

import moeabench as mb
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from unittest.mock import patch


def signature(value):
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def rng_signature(kwargs):
    rng = kwargs.get("random_state")
    if rng is None:
        return None
    try:
        payload = json.dumps(rng.bit_generator.state, sort_keys=True, default=str)
    except (TypeError, AttributeError):
        return type(rng).__name__
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    events = []
    original_crossover = SBX._do
    original_mutation = PolynomialMutation._do

    def crossover(self, problem, X, **kwargs):
        before = signature(X)
        rng_before = rng_signature(kwargs)
        produced = original_crossover(self, problem, X, **kwargs)
        events.append({
            "operator": "SBX", "step": len(events),
            "input": before, "output": signature(produced),
            "rng_before": rng_before, "rng_after": rng_signature(kwargs),
        })
        return produced

    def mutation(self, problem, X, **kwargs):
        before = signature(X)
        rng_before = rng_signature(kwargs)
        produced = original_mutation(self, problem, X, **kwargs)
        events.append({
            "operator": "polynomial_mutation", "step": len(events),
            "input": before, "output": signature(produced),
            "rng_before": rng_before, "rng_after": rng_signature(kwargs),
        })
        return produced

    with patch.object(SBX, "_do", new=crossover), \
         patch.object(PolynomialMutation, "_do", new=mutation):
        exp = mb.experiment(
            mop=mb.mops.DPF1(M=3, D=2),
            moea=mb.moeas.NSGA3(population=24, generations=12, seed=12345)
        )
        exp.run(repeat=1, silent=True)

    result = {
        "final": signature(np.asarray(exp.front())),
        "event_count": len(events),
        "events": events,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
