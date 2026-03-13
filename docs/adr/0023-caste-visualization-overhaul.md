<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# 23. Caste Visualization Overhaul

Date: 2026-02-02

## Status

Accepted

> [!NOTE]
> Historical naming note: this ADR discusses the visualization now exposed canonically as `mb.view.strata(...)`. Earlier names such as `strat_caste` and `mb.view.caste(...)` are legacy terminology.

## Context

The original strata visualization provided a basic view of rank quality vs. density but suffered from critical limitations:
1.  **Metric Ambiguity**: It was unclear whether the y-axis represented aggregated or individual quality.
2.  **Visual Clutter**: Default annotations were often overlapping and hard to read.
3.  **Lack of Robustness Insight**: It did not explicitly show stochastic variance across multiple runs.
4.  **No Programmatic Data Access**: Users had to rely on the plot labels to read values.

## Decision

We have decided to completely overhaul the strata visualization function (formerly prototyping as `strat_caste2`) and deprecate the old implementation.

### 1. Parametric Modes
The new strata visualization introduces a `mode` parameter:
*   **`mode='collective'` (Default)**: Visualizes the **Gross Domestic Product (GDP)** of each rank across multiple runs. This measures the algorithm's **stochastic robustness**—a short box means high reliability.
*   **`mode='individual'`**: Visualizes the **Per Capita** merit distribution of individual solutions. This measures the algorithm's **internal diversity**.

### 2. Scientific Aesthetic
*   **Clean Annotations**: All alphanumeric prefixes (`n:`, `q:`) were removed. The plot now uses a high-density numeric-only style aligned with Tufte's principles of data-ink ratio.
*   **Statistical Alignment**: Secondary labels now align with statistical whiskers ($1.5 \times IQR$) rather than absolute outliers, providing a more rigorous view of the distribution.

### 3. Programmatic Access (`StrataSummary`)
We introduced a method-based API on `LayerResult` to allow uniform access to the visualized data:
```python
res.strata_summary().n(1)  # Population count
res.strata_summary().q(1)  # Median quality
```

## Consequences

*   **Breaking Change**: The strata-view signature changed, and callers should use the canonical `mb.stats.strata(...) -> mb.view.strata(...)` flow.
*   **Improved Rigor**: The distinction between 'performance' (individual) and 'robustness' (collective) is now explicit in the library's visual vocabulary.
*   **Data Portability**: Users can now easily export strata statistics to LaTeX/Pandas without image processing.
