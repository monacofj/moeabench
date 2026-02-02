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

1.  **HV_raw (Physical Volume)**: The absolute volume dominated by the solution set relative to a specified reference point (e.g., $(1.1, 1.1, 1.1)$).
2.  **HV_ratio (Coverage)**: The fraction of the **Reference Box** (defined from Ideal to Nadir+Offset) covered by the solution. Formula: $HV_{raw} / \text{Volume}(\text{RefBox})$. This indicates how well the algorithm explored the defined search area.
3.  **HV_rel (Convergence)**: The fraction of the **Ground Truth Hypervolume** achieved by the solution. Formula: $HV_{sol} / HV_{GT}$. This is the primary indicator of how close the algorithm got to the optimal front.

## Consequences

### Positive
*   **Semantic Clarity**: Distinguishes between "How much did we find?" (Ratio) and "How close are we to the truth?" (Rel).
*   **Artifact Resolution**: Values > 100% in `HV_rel` are now explicitly identified as "Sampling Saturation" rather than a bug in the volume engine.
*   **Scientific Rigor**: Provides a more nuanced view of algorithm performance, especially in many-objective scenarios.

### Negative
*   **Report Complexity**: Baseline reports now contain three columns for Hypervolume instead of one.
