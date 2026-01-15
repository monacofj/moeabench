# ADR 0010: Rich Result Objects and Narrative Reporting

**Status**: Accepted  
**Date**: 2026-01-15

## Context

Statistical analysis in multi-objective optimization often yields complex data (p-values, effect sizes, distributions, quality profiles). Traditionally, these are returned as raw floats or dictionaries, forcing the user to manually print and interpret them. This creates boilerplate in scripts and hides the analytical story of the data.

## Decision

We will implement a unified "Rich Result" system for all tools in the `mb.stats` module.

### 1. The `StatsResult` Interface
All statistical functions (e.g., `mb.stats.strata`, `mb.stats.mann_whitney`, `mb.stats.attainment_diff`) will return objects inheriting from a base `StatsResult` class.

### 2. Narrative Reporting (`.report()`)
Every result object provides a `.report()` method. This method generates a formatted, human-readable summary that includes:
- Summary of the inputs.
- Key statistics.
- **Diagnosis**: A high-level interpretation (e.g., "High Selection Pressure", "Significant difference favoring Algorithm A").

### 3. Lazy Evaluation
To maintain high performance and memory efficiency, result objects use **Lazy Evaluation** (via `@cached_property`). Properties like effect sizes, quality profiles, or diagnostic strings are only calculated at the exact moment the user (or the report) requests them. Once calculated, they are cached for future access.

### 4. Direct Programmatic Access
Every metric displayed in the narrative report must be accessible as a first-class property of the result object (e.g., `res.p_value`, `res.selection_pressure`). This ensures the library is equally powerful for automated batch processing and interactive exploration.

## Consequences

- **Analytical Value**: Example scripts become much cleaner, using `print(res.report())` instead of complex manual formatting.
- **API Consistency**: Users can expect a similar "feeling" when using any statistical tool in MoeaBench.
- **Efficiency**: Large experiments can return result objects quickly, with expensive secondary metrics deferred until needed.
