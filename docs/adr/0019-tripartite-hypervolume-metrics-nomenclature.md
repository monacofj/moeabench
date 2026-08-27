<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0019: Tripartite Hypervolume Metrics Nomenclature

## Status

Partially Superseded by [ADR 0047](0047-conventional-hypervolume-relative-scaling.md). Implemented in v0.7.6 and evolved in v0.11.2 through [ADR 0035](0035-triple-mode-hypervolume-reporting.md).

## Context

Previous versions of moeabench reported a single "Hypervolume" value. During the v0.7.6 audit, this was found to be ambiguous, leading to confusion when values exceeded 100% (due to discrete sampling) or when different reference points were used. Peer reviewers requested a clearer distinction between physical volume, coverage of the reference box, and convergence to the ground truth.

## Technical Decision

We decided to decompose the Hypervolume metric into three distinct physical interpretations:

1.  **H_raw (Contextual Volume)**: The volume dominated by the solution set within the selected normalization context. The context is self-derived when `ref` is omitted and externally fixed when `ref` is supplied.
    *   *Unit*: normalized objective-space volume in that context.
2.  **H_rel (Competitive Efficiency)**: Scaled by the session's maximum found volume. Forces a 1.0 ceiling for the current winner. (Formerly `H_ratio`).
    *   *Interpretation*: 0.0 to 1.0 (relative to competition).
3.  **H_abs (Theoretical Optimality)**: Scaled by the volume of the mathematical Ground Truth.
    *   *Formula*: $\frac{H_{sol}}{H_{GT}}$.
    *   *Interpretation*: 1.0 means mathematical perfection. (Formerly `H_rel`).

> [!NOTE]
> **v0.11.x Evolution**: Starting with v0.11, the nomenclature was refined for scientific clarity. `H_ratio` was renamed to `H_rel` (Relative Efficiency), and the old `H_rel` (Truth) was renamed to `H_abs` (Absolute Optimality). See **[ADR 0035](0035-triple-mode-hypervolume-reporting.md)** for details.

## Consequences

### Positive
*   **Semantic Clarity**: Distinguishes between "How much did we find?" (Ratio) and "How close are we to the truth?" (Rel).
*   **Artifact Resolution**: Values > 100% in `HV_rel` are now explicitly identified as "Sampling Saturation" rather than a bug in the volume engine.
*   **Scientific Rigor**: Provides a more nuanced view of algorithm performance, especially in many-objective scenarios.

### Negative
*   **Report Complexity**: Baseline reports now contain three columns for Hypervolume instead of one.

## Current Status Note

This ADR is retained as the historical origin of the tripartite nomenclature. Its session-maximum definition and `1.0` ceiling for `H_rel` are no longer the current operational contract. [ADR 0047](0047-conventional-hypervolume-relative-scaling.md) defines the current relative-scaling semantics, including explicit-reference denominators and unclipped values above `1.0`. The GT-normalized perspective described historically as convergence to truth is now canonically named `H_abs`.
