<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0008: Population Strata (Rank Distribution)

**Status**: Accepted  
**Date**: 2026-01-14

> [!NOTE]
> Historical ADR. The analytical idea remains accepted, but the canonical public API was later refined by [ADR 0039](./0039-canonical-api-and-compare-semantics-v0.14.0.md) and [ADR 0040](./0040-canonical-view-inputs-and-no-compatibility-shims.md).
> The modern public surface is `mb.stats.ranks`, `mb.stats.caste`, `mb.stats.tiers` with `mb.view.ranks`, `mb.view.caste`, and `mb.view.tiers`.

## Context

Typical Multi-Objective Evolutionary Algorithm (MOEA) analysis focuses almost exclusively on the final Pareto Front (Rank 1). While Hypervolume and IGD provide a snapshot of success, they often fail to explain why an algorithm converged or stalled. In many-objective scenarios, or when selection pressure is unbalanced, the internal health of the remaining non-domination layers remains a black box.

## Decision: The "Strata" Narrative

We decided to implement **Population Strata** as a first-class structural diagnostic. The original implementation centered on `mb.stats.strata`; the current canonical public API preserves the same analytical idea through explicit result families: `mb.stats.ranks`, `mb.stats.caste`, and `mb.stats.tiers`.

### 1. The "Onion Peeling" Core
To support this, we extended our Non-Dominated Sorting (NDS) logic. Beyond finding the first front, we implemented a recursive "layer-peeling" method (`Population.stratify()`). 

- **Rank 1**: The standard Pareto Front.
- **Rank 2**: The Pareto Front of the population remaining after Rank 1 is removed.
- **Rank 3+**: Subsequent layers peeled from the population.

### 2. Measuring the Invisible: Selection Pressure
...

### 3. Symmetric Quantitative Comparison: Earth Mover's Distance (EMD)
Comparing two histograms is often reduced to visual inspection. To provide scientific rigor and maintain API symmetry, we chose to use a functional, symmetric **`mb.stats.emd(strat1, strat2)`** (Earth Mover's Distance / Wasserstein distance).

Unlike an asymmetric class method, this symmetric function treats both algorithms as equal samples, calculating the "work" required to transform one dominance profile into the other.

## Rationale: Diagnostics as a Research Foundation

This implementation aligns with the moeabench commitment to "Analysis-First" research. By exposing the sub-Pareto layers, we empower researchers to detect issues like "Dominance Resistance" in many-objective problems or "Loss of Diversity" in stagnant populations—insights that are completely invisible to standard metrics like Hypervolume.

## Consequences
- **Positive**: New diagnostic depth for multi-run experiments.
- **Positive**: Quantitative comparison of selection pressure and dominance profiles.
- **Negative**: Full stratification is $O(N^2)$, which can be slow for massive populations (though mitigated by our chunked vectorization).
- **Neutral**: Adds `scipy.stats` dependencies (already present in the library).
