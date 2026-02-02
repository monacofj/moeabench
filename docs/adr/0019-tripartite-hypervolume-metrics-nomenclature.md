<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0019: Tripartite Hypervolume Metrics Nomenclature

## Status

Accepted (Implemented in v0.7.6)

## Context

Previous versions of MoeaBench reported a single "Hypervolume" value. During the v0.7.6 audit, this was found to be ambiguous, leading to confusion when values exceeded 100% (due to discrete sampling) or when different reference points were used. Peer reviewers requested a clearer distinction between physical volume, coverage of the reference box, and convergence to the ground truth.

## Technical Decision

We decided to decompose the Hypervolume metric into three distinct physical interpretations:

1.  **H_raw (Physical Volume)**: The absolute volume dominated by the solution set relative to the reference point $\vec{r} = [1.1, \dots, 1.1]$.
    *   *Unit*: $Obj_1 \cdot Obj_2 \cdot \dots \cdot Obj_M$.
2.  **H_ratio (Exploration/Coverage)**: Scaled by the volume of the search area (Reference Box).
    *   *Formula*: $\frac{H_{raw}}{V_{RefBox}}$ where $V_{RefBox} = \prod (r_i - Ideal_i)$.
    *   *Interpretation*: 0.0 (nothing found) to 1.0 (perfect coverage of the box).
3.  **H_rel (Convergence/Truth)**: Scaled by the volume of the optimal set.
    *   *Formula*: $\frac{H_{sol}}{H_{GT}}$.
    *   *Interpretation*: 1.0 means the algorithm matched the Ground Truth hypervolume. > 1.0 indicates sampling saturation (filling GT gaps).

## Consequences

### Positive
*   **Semantic Clarity**: Distinguishes between "How much did we find?" (Ratio) and "How close are we to the truth?" (Rel).
*   **Artifact Resolution**: Values > 100% in `HV_rel` are now explicitly identified as "Sampling Saturation" rather than a bug in the volume engine.
*   **Scientific Rigor**: Provides a more nuanced view of algorithm performance, especially in many-objective scenarios.

### Negative
*   **Report Complexity**: Baseline reports now contain three columns for Hypervolume instead of one.
