<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0020: Standardized Reference Point Strategy for Normalization

## Status

Accepted (Implemented in v0.7.6)

## Context

MoeaBench normalizes all search data to a unit hypercube $[0, 1]^M$ based on the joint range of the Ground Truth and all algorithm populations. However, calculating Hypervolume at the exact boundary ($1.1$ or $1.0$) can lead to numerical artifacts where points on the boundary are ignored or contribute zero volume. Peer review suggested a consistent offset to ensure stability.

## Technical Decision

We decided to standardize the **Reference Point** for Hypervolume calculation at **1.1** (Nadir + 10% offset) in all normalized calibration pipelines. 

1.  **Normalization Protocol**: Data is linearly scaled such that the joint global Ideal maps to $0.0$ and the joint global Nadir maps to $1.0$.
2.  **Reference Point**: Metrics are computed relative to $\vec{r} = [1.1, 1.1, \dots, 1.1]$.
3.  **Rationale**: This ensures that even "worst-case" solutions at the Nadir contribute to the hypervolume and that the reference box is large enough to encapsulate the entire search front without boundary effects.

## Consequences

### Positive
*   **Consistency**: All algorithms in the calibration report are compared against the same physical volume ($1.1^M$).
*   **Stability**: Eliminates edge cases in the WFG (exact) hypervolume algorithm.
*   **Clarity**: The reference box volume is explicitly defined as $1.1^M$, making `HV_ratio` calculations deterministic.

### Negative
*   **Value Shift**: Raw HV values will increase relative to a 1.0 reference, but `HV_rel` (convergence to GT) remains invariant.
