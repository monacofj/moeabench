<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# 23. Caste Visualization Overhaul

Date: 2026-02-02

## Status

Accepted

## Context

The original `strat_caste` visualization provided a basic view of rank quality vs. density but suffered from critical limitations:
1.  **Metric Ambiguity**: It was unclear whether the y-axis represented aggregated or individual quality.
2.  **Visual Clutter**: Default annotations were often overlapping and hard to read.
3.  **Lack of Robustness Insight**: It did not explicitly show stochastic variance across multiple runs.
4.  **No Programmatic Data Access**: Users had to rely on the plot labels to read values.

## Decision

We have decided to completely overhaul the `strat_caste` function (formerly prototyping as `strat_caste2`) and deprecate the old implementation.

### 1. Parametric Modes
The new `strat_caste` introduces a `mode` parameter:
*   **`mode='collective'` (Default)**: Visualizes the **Gross Domestic Product (GDP)** of each rank across multiple runs. This measures the algorithm's **stochastic robustness**—a short box means high reliability.
*   **`mode='individual'`**: Visualizes the **Per Capita** merit distribution of individual solutions. This measures the algorithm's **internal diversity**.

### 2. Scientific Aesthetic
*   **Clean Annotations**: All alphanumeric prefixes (`n:`, `q:`) were removed. The plot now uses a high-density numeric-only style aligned with Tufte's principles of data-ink ratio.
*   **Statistical Alignment**: Secondary labels now align with statistical whiskers ($1.5 \times IQR$) rather than absolute outliers, providing a more rigorous view of the distribution.

### 3. Programmatic Access (`CasteSummary`)
We introduced a method-based API on `StratificationResult` to allow uniform access to the visualized data:
```python
res.caste_summary().n(1)  # Population count
res.caste_summary().q(1)  # Median quality
```

## Consequences

*   **Breaking Change**: The signature of `strat_caste` has changed. Legacy code should use `strat_caste_deprecated` or update to the new API.
*   **Improved Rigor**: The distinction between 'performance' (individual) and 'robustness' (collective) is now explicit in the library's visual vocabulary.
*   **Data Portability**: Users can now easily export caste statistics to LaTeX/Pandas without image processing.
