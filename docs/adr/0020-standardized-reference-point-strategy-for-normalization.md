<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0020: Standardized Reference Point Strategy for Normalization

## Status

Accepted (Implemented in v0.7.6)

## Context

moeabench normalizes Hypervolume data to a unit hypercube $[0, 1]^M$ using one selected context. Without `ref`, that context is derived collectively from the evaluated experiment's runs. With `ref`, ideal and nadir come exclusively from the external reference. Calculating Hypervolume at the exact nadir boundary can otherwise cause points on the boundary to contribute zero volume.

## Technical Decision

We decided to standardize the **Reference Point** for Hypervolume calculation at **1.1** (Nadir + 10% offset) in all normalized calibration pipelines. 

1.  **Normalization Protocol**: Data is linearly scaled so the selected context's ideal maps to $0.0$ and its nadir maps to $1.0$.
2.  **Reference Point**: Metrics are computed relative to $\vec{r} = [1.1, 1.1, \dots, 1.1]$.
3.  **Rationale**: This ensures that solutions at the selected nadir contribute to Hypervolume. Evaluated points never expand externally supplied bounds; points beyond the normalized reference point contribute no volume, while better-than-ideal points may contribute outside the nominal unit cube.

## Consequences

### Positive
*   **Consistency**: All algorithms in the calibration report are compared against the same physical volume ($1.1^M$).
*   **Stability**: Eliminates edge cases in the WFG (exact) hypervolume algorithm.
*   **Clarity**: The reference box volume is explicitly defined as $1.1^M$, making `HV_ratio` calculations deterministic.

### Negative
*   **Value Shift**: Raw HV values will increase relative to a 1.0 reference, but `HV_rel` (convergence to GT) remains invariant.
