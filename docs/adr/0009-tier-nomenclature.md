<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2026 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0009: Tier-based Competitive Analysis

## Status
Accepted

## Context
During the evolution of the library's competitive analysis features (designed to compare two algorithms), the nomenclature was inconsistent. Terms like "Arena", "Dominance Duel", and "Caste" were used interchangeably in prototypes. This fragmentation caused confusion in the API (`mb.arena()` vs `mb.domplot()`) and in the documentation.

## Decision
We have standardized all competitive and hierarchical analysis under the **"Tier"** nomenclature.

1.  **Standardized Module**: The core comparison logic is now encapsulated in `mb.stats.tier()` and visualized via `mb.view.tierplot()`.
2.  **Removal of Aliases**: All legacy aliases (`arena`, `domplot`) have been removed from the public namespace to enforce a clean, unambiguous API.
3.  **Metrics Standardization**: The primary metrics for competitive analysis are now formally defined as **"Dominance Ratio"** and **"Displacement Depth"**.
4.  **Tabula Rasa Policy**: We performed a "Tabula Rasa" cleanup, removing "formerly known as" notes to present a cohesive, finalized architecture.

## Consequences
- **Positive**: API predictability is greatly improved.
- **Positive**: Documentation is now concise and free of legacy jargon.
- **Negative**: Breaking change for users who relied on the early "Arena" prototype nomenclature.
- **Neutral**: The term "Tier" better reflects the sorted, hierarchical nature of the dominance levels compared to the more combative "Arena".
