<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0004: Analysis-First Specialization and "Smart Stats"

**Status**: Accepted  
**Date**: 2026-01-14

## Context
In the Multi-Objective Evolutionary Algorithm (MOEA) research space, the market is saturated with solvers but starving for **analysis tools**. Most researchers follow a standard workflow:
1. Run Algorithm A and Algorithm B 30 times each.
2. Calculate Hypervolume for 60 history files.
3. Extract the final generation of each run.
4. Run a Mann-Whitney U test manually in another script (or even Excel).
5. Attempt to plot an Attainment Surface to see the distribution.

This workflow is fragmented, error-prone, and requires a lot of "boilerplate" code to move data between optimization and statistics libraries.

## Decision
We decided to specialize MoeaBench as an **Analysis-First platform**. The library is designed to facilitate the "End Game" of a research paper: the statistical validation and visualization of results.

### Implementation Pillars
1.  **Smart Stats API**: We integrated statistical tests (`mann_whitney`, `ks_test`, `a12`) directly into the `Experiment` and `MetricMatrix` objects. The library knows how to extract the correct distribution from an experiment automatically, removing the need for manual data extraction.
2.  **Attainment as a First-Class Citizen**: We implemented Empirical Attainment Functions (EAF) and Surfaces (`attainment`, `attainment_diff`) natively. This allows users to visualize not just the "best" front, but the 5%, 50%, and 95% boundaries of their algorithm's performance distribution.
3.  **Visualization Integration**: Plots in MoeaBench (via `spaceplot` and `timeplot`) are "Smart Aware." They automatically detect if they are plotting a single solution set, a time series of metrics, or a statistical attainment surface, adjusting labels and legends accordingly.

## Rationale
By focusing on analysis, MoeaBench becomes more than just another optimizer—it becomes a laboratory. We prioritize "Statistical Rigor" by automating the steps that researchers often skip or perform incorrectly (like ensuring a common reference point for Hypervolume before comparison).

## Consequences
- **Positive**: Dramatically reduces the time from "optimization finished" to "ready for publication."
- **Positive**: Provides a unique value proposition that differentiates MoeaBench from general-purpose libraries like Pymoo or DEAP.
- **Negative**: The library is more opinionated about data structure. To use these tools, users must follow the `Experiment`/`Run` data model.
- **Neutral**: Adds standard scientific dependencies like `scipy` and `pandas`.
