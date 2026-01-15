<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0002: The "Maestro" Algorithm Wrapper Strategy

**Status**: Accepted  
**Date**: 2026-01-14

## Context
A recurring trap in Multi-Objective Evolutionary Algorithm (MOEA) development is the "re-implementation loop." Implementing algorithms like NSGA-III, MOEA/D, or SPEA2 from scratch is a significant engineering undertaking. Each algorithm has subtle nuances—such as niche management, reference point updates, or specific crossover constraints—that are easy to get wrong and difficult to validate. 

Spending our engineering effort on re-implementing these solvers would divert resources away from the true goal of MoeaBench: providing a superior platform for **experimentation and analysis**.

## Decision
We decided to adopt a **"Maestro" strategy**, where MoeaBench acts as a high-level conductor for mature, industry-standard optimization engines. 

Instead of writing our own solvers, we developed a robust **Wrapper Pattern** (`BaseMoea` and `BaseMoeaWrapper`):
1.  **Integration of Pymoo**: We leverage Pymoo as our primary engine due to its rigor, extensive documentation, and validation by the research community.
2.  **Integration of DEAP**: We maintain support for logic-heavy genetic frameworks like DEAP when specific procedural control is needed.
3.  **Encapsulation**: The user never interacts with the underlying engine directly. MoeaBench translates a simple set of parameters (`population`, `generations`, `seed`) into the complex engine-specific configurations.
4.  **Standardization**: MoeaBench handles the "boring" parts of the search—seed incrementing across runs, progress bar reporting, and result extraction—ensuring that output format remains identical regardless of whether Pymoo or DEAP is used.

## Rationale: Why not implement from scratch?
- **Validation**: Established libraries have been tested against thousands of MOPs. A custom implementation would require massive benchmarking just to prove it matches the original papers.
- **Focus**: MoeaBench's unique value is the "post-processing" (metrics, stats, attainment). By externalizing the search engine, we can dedicate 100% of our focus to these analytical tools.

## The Distinction: What we Hand-Craft instead
While we wrap the *Searching Engines* (MOEAs), we deliberately re-implement the following components for performance and specialized utility:
1.  **Multi-objective Problems (MOPs)**: We re-implemented standard benchmarks (e.g., DTLZ) using NumPy broadcasting. This ensures that population evaluations are much faster than the loop-based implementations often found in generic libraries.
2.  **Attainment Statistics**: Empirical Attainment Functions (EAF) and surface calculations are implemented from scratch in MoeaBench. This provides a deep, multi-run statistical view that is absent in most optimization-only engines.
3.  **Smart Metric Selectors**: The logic for robustly extracting metrics from complex history objects (via `MetricMatrix`) is a native feature designed for research workflows.

## Consequences
- **Positive**: Instant access to state-of-the-art algorithms that are guaranteed to work correctly.
- **Positive**: Users can switch between algorithms from different libraries using a single, consistent API.
- **Positive**: Dramatically reduced maintenance surface; we only need to maintain the "bridge," not the "engine."
- **Negative**: Adds heavy dependencies (principally `pymoo`), which can complicate installation in some environments.
- **Neutral**: The search performance (speed) is capped by the efficiency of the external library, though this is usually optimal as these libraries are already written in C/Cython.
