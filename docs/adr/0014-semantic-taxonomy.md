<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2026 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0014: Semantic Taxonomy (Performance vs. Topology)

## Status
Accepted

## Context
As MoeaBench matured from a collection of scripts into a benchmark library for researchers, the function names in `mb.stats` and `mb.view` remained tied to their underlying implementation details (e.g., `mann_whitney`, `ks_test`, `a12`, `attainment`). This technical-centric naming required users to possess prior knowledge of the statistical tests before understanding the investigative purpose of the tool. Furthermore, the lack of symmetry between statistical analysis (`stats`) and its corresponding visualization (`view`) created cognitive overhead.

## Decision
We have implemented a **Semantic Taxonomy** that reorganizes the library's analytical capabilities into two distinct domains based on the researcher's intent:

1.  **Performance Domain (`perf_*`)**: Focuses on "who is better?" by evaluating scalar metrics (HV, IGD) through statistical inference.
    *   `perf_evidence`: Statistical significance (Mann-Whitney U).
    *   `perf_prob`: Win probability and effect size (Vargha-Delaney A12).
    *   `perf_dist`: Distribution shape comparison (Kolmogorov-Smirnov).

2.  **Topologic Domain (`topo_*`)**: Focuses on "what was found?" by evaluating the spatial distribution and coverage of solutions.
    *   `topo_dist`: Multi-axial distribution matching (KS, Anderson-Darling, or EMD).
    *   `topo_attain`: Empirical Attainment Functions (EAF).
    *   `topo_gap`: Spatial coverage differences (EAF Difference).

3.  **Visual Parity**: Plotters in `mb.view` now share the same semantic names (e.g., `mb.view.topo_dist`), ensuring that the research outcome is conceptually linked to the visualization method.

4.  **Clean Slate Enforcement**: To prevent technical debt and ensure absolute consistency, all legacy aliases and deprecated shortcuts have been permanently removed. v0.6.0 introduces a clean break with the new taxonomy.

## Consequences
- **Positive**: The API is now self-documenting for researchers ("I want win probability" $\to$ `perf_prob`).
- **Positive**: Clear separation between objective quality analysis (Performance) and search behavior analysis (Topology).
- **Positive**: Strict API symmetry between stats and view modules improves predictability.
- **Negative**: Breaking change that requires users to update their scripts.
- **Positive**: Explicit documentation of mathematical tests within the API ensures scientific rigor.
