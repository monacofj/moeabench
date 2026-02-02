<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0022: EMD as Primary Diagnostic for Degenerate Manifolds

## Status

Accepted (Implemented in v0.7.6)

## Context

On degenerate problems like DPF3 ($x^{100}$ mapping), traditional metrics like IGD can misrepresent algorithm performance. An algorithm can clumping its population in the center of the front (low distance error) while missing $2/3$ of the manifold's extent. 

## Technical Decision

We decided to promote the **Earth Mover's Distance (EMD)** as the primary diagnostic metric for **Topological Diversity**.

1.  **Metric Precedence**: In the calibration report, when IGD is low but EMD is high, the EMD signal takes precedence in diagnosing algorithm failure.
2.  **Topological Validation**: We established that EMD correctly identifies "Boundary Recession" (loss of manifold extents) even when individual points are highly converged (low IGD).
3.  **Verification**: Validated via controlled groups that EMD is insensitive to cardinality differences but highly sensitive to clumping and holes in the objective 1D/2D manifolds.

## Consequences

### Positive
*   **Superior Diagnosis**: Reveals failures that proximity-based metrics (IGD, GD) cannot see.
*   **Guidance for Optimization**: Provides a clear signal for the need for better weight distribution (MOEA/D) or diversity maintenance (NSGA-II) on degenerate fronts.

### Negative
*   **Complexity**: EMD is computationally more expensive than IGD, making it more suitable for post-processing/audit rather than real-time search guidance.
