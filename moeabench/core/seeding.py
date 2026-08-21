# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Deterministic seed allocation for independent stochastic components."""

import hashlib
from numbers import Integral


UINT32_MAX = 2**32 - 1


def validate_seed(seed, *, name="seed", max_value=None):
    """Validate and return a non-negative integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"{name} must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"{name} must be non-negative")
    if max_value is not None and seed > max_value:
        raise ValueError(f"{name} must be at most {max_value}")
    return seed


def derive_component_seed(master_seed, component):
    """Derive a stable 32-bit component seed without consuming an RNG."""
    master_seed = validate_seed(master_seed, name="master_seed")
    if not isinstance(component, str):
        raise TypeError("component must be a string")
    if not component:
        raise ValueError("component must be a non-empty string")

    payload = f"moeabench-seed-v1\0{master_seed}\0{component}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big")
    if derived == master_seed:
        derived = (derived + 1) % 2**32
    return derived
