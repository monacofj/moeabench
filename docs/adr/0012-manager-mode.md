<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2026 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0012: Manager Mode: Experiment Delegation Semantics

## Status
Accepted

## Context
In version 0.3.0, the `experiment` object acted as a split-personality manager. Some methods (like `exp.pop()`) operated on the aggregation of all runs (the "cloud"), while delegation methods (like `exp.front()` or `exp.set()`) were hardcoded to point only to the `last_run`. This inconsistency made the manager-level API unpredictable and chemically distinct from the individual `Run` objects.

## Decision
We implemented the **"Manager Mode"** architecture, unifying the perspective of the `experiment` object.

1.  **Cloud-Centric Delegation**: All methods called directly on the `experiment` object (`front()`, `set()`, `non_dominated()`, `dominated()`, etc.) now operate on the **aggregated cloud** of all runs by default.
2.  **Separation of Concerns**: 
    *   **Manager (Experiment)**: Provides the global, statistical perspective of the research.
    *   **Trajectory (Run)**: Provides surgical access to individual stochastic paths.
3.  **Terminology Shift**: We replaced the jargon "Surgical Access" with **"Single-run access"** to describe explicit access to individual runs via indexing (`exp[i]`) or `.last_run`.
4.  **Deprecation of "Super" Methods**: Methods like `superfront()` and `superset()` were deprecated, as their behavior is now the default for `front()` and `set()`.

## Consequences
- **Positive**: The API is now semantically consistent: `Experiment` is always the global view.
- **Positive**: Simplified idiom for users: `mb.view.spaceplot(exp)` automatically shows the aggregated elite.
- **Positive**: Structural parity between `Population` and `JoinedPopulation` (cloud) through shared properties (`objs`, `vars`).
- **Negative**: Slight performance overhead when calling `exp.front()` due to the multi-run aggregation logic.
