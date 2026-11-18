# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only NSGA3/DPF1 variation signature probe.

Wrap the real pymoo SBX and PolynomialMutation operators, record inputs,
outputs and available RNG state, but return their unmodified results.
No baseline, production code, or regression test is changed.
"""
import hashlib
import json
import copy
import math

import numpy as np

import moeabench as mb
from pymoo.operators.crossover.sbx import SBX
import pymoo.operators.crossover.sbx as sbx_module
from pymoo.core.problem import Problem
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
        first_rng_state = (copy.deepcopy(kwargs["random_state"].bit_generator.state)
                           if not events and kwargs.get("random_state") is not None
                           else None)
        produced = original_crossover(self, problem, X, **kwargs)
        event = {
            "operator": "SBX", "step": len(events),
            "input": before, "output": signature(produced),
            "rng_before": rng_before, "rng_after": rng_signature(kwargs),
        }
        if not events:
            # Preserve a standalone, replayable first SBX input and RNG state.
            event["first_input_values"] = np.asarray(X).tolist()
            event["first_rng_state"] = first_rng_state
            event["first_output_values"] = np.asarray(produced).tolist()
            if kwargs.get("random_state") is not None:
                # The original state is supplied separately by the outer caller
                # below; this post-call state is only supplementary evidence.
                event["rng_state_after_full"] = copy.deepcopy(
                    kwargs["random_state"].bit_generator.state
                )
        events.append(event)
        return produced

    def mutation(self, problem, X, **kwargs):
        before = signature(X)
        rng_before = rng_signature(kwargs)
        produced = original_mutation(self, problem, X, **kwargs)
        event = {
            "operator": "polynomial_mutation", "step": len(events),
            "input": before, "output": signature(produced),
            "rng_before": rng_before, "rng_after": rng_signature(kwargs),
        }
        if len(events) == 1:
            event["first_output_values"] = np.asarray(produced).tolist()
        events.append(event)
        return produced

    with patch.object(SBX, "_do", new=crossover), \
         patch.object(PolynomialMutation, "_do", new=mutation):
        exp = mb.experiment(
            mop=mb.mops.DPF1(M=3, D=2),
            moea=mb.moeas.NSGA3(population=24, generations=12, seed=12345)
        )
        exp.run(repeat=1, silent=True)

    # Independent replay of the *first* SBX call from archived input
    # and RNG state; no evolution and no production algorithm changes.
    first = events[0]
    first_input = np.asarray(first["first_input_values"], dtype=float)
    before_rng = first["first_rng_state"]
    replay_problem = Problem(
        n_var=first_input.shape[2], n_obj=3,
        xl=np.zeros(first_input.shape[2]), xu=np.ones(first_input.shape[2])
    )

    def replay(*, use_scalar_power=False):
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(before_rng)
        crossover = SBX(prob=1.0, eta=15)
        if use_scalar_power:
            def scalar_power(a, b):
                # Diagnostic comparison with the process scalar libm.
                return np.frompyfunc(math.pow, 2, 1)(a, b).astype(float)
            with patch.object(sbx_module.np, "power", side_effect=scalar_power):
                out = crossover._do(replay_problem, first_input.copy(), random_state=rng)
        else:
            out = crossover._do(replay_problem, first_input.copy(), random_state=rng)
        return np.asarray(out)

    numpy_replay = replay()
    scalar_replay = replay(use_scalar_power=True)
    measured = np.asarray(first["first_output_values"])
    result = {
        "final": signature(np.asarray(exp.front())),
        "event_count": len(events),
        "events": events,
        "sbx_replay": {
            "replays_recorded_input_and_rng": bool(np.array_equal(numpy_replay, measured)),
            "numpy": signature(numpy_replay),
            "scalar_math_pow": signature(scalar_replay),
            "max_abs_delta_scalar_vs_numpy": float(np.max(np.abs(scalar_replay - numpy_replay))),
        }
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
