<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2026 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0013: Centralized Visual Identity (Ocean Palette)

## Status
Accepted

## Context
Initial versions of MoeaBench relied on default Matplotlib and Plotly styles, leading to visual fragmentation and a "generic" look. Scientific plots lacked a cohesive brand identity, and custom color cycles were often hardcoded within individual plotting functions, making them difficult to maintain or override.

## Decision
We established a centralized visual identity system based on the **"Ocean Palette"**.

1.  **Branding Palette**: Defined a specific 9-color categorical palette (Indigo, Emerald Green, Soft Plum, etc.) optimized for high contrast and scientific clarity.
2.  **Centralized Style Manager**: Implemented `MoeaBench/view/style.py` as a single source of truth for all visual tokens.
3.  **Automatic Backend Synchronization**: The `apply_style()` function, called on `mb.view` import, automatically configures:
    *   **Matplotlib**: Sets `axes.prop_cycle` and grid aesthetics globally.
    *   **Plotly**: Registers a custom `moeabench` template as the default.
4.  **Aesthetic Enforcement**: Refactored visualizations (like `casteplot` and `tierplot`) to remove hardcoded colormaps in favor of the standard property cycle (`C0`, `C1`, etc.).

## Consequences
- **Positive**: A consistent, premium visual identity across all "Scientific Perspectives".
- **Positive**: Maintenance effort is reduced—changing one palette token updates all plots.
- **Positive**: Seamless transition between static (Matplotlib) and interactive (Plotly) modes with matching colors.
- **Negative**: Users who want to strictly use their own global `rcParams` must manually override the MoeaBench style after import.
