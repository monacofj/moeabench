# ADR 0035: Dual-Mode Hypervolume Reporting (Raw vs Ratio)

## Status
Accepted

## Context
When calculating the Hypervolume of multiple algorithms/experiments in a dynamic environment (without a pre-defined maximum bounding box or Ground Truth), the Bounding Box (BBox) must dynamically expand to accommodate the worst solutions found by any algorithm in the set.

Previously, `MoeaBench` implemented a dynamic normalization step where the calculated absolute volume of each algorithm was subsequently divided by the **maximum volume found in the session**. 

This forced the best algorithm to naturally hit a `1.0` ceiling. However, this caused a significant "Perspective Illusion":
1. When Algorithm A (highly stable, but perhaps suboptimal) is evaluated alone, its maximum performance is `1.0`. Due to internal variances, its average performance across runs might be reported as, e.g., `0.77`.
2. When Algorithm B (a poorly performing algorithm that explores very bad areas of space) is added to the comparison, the BBox expands massively.
3. Within this massive new BBox, the *volume* occupied by Algorithm A's average runs and Algorithm A's best runs is almost identical compared to the disastrous volume occupied by Algorithm B. 
4. The post-processing division (normalizing to `1.0`) suddenly reports Algorithm A's average as `0.91`, shifting the reported metric solely because a bad neighbor altered the scale of the box.

This mathematical behavior violates the principle of "Immutable Evaluation" for physical metrics while satisfying the desire for "Competitive Efficiency Rankings".

## Decision
To resolve the contradiction between physical measurement and competitive ranking, we decided to split the Hypervolume calculation into two distinct, user-selectable modes within the core algorithmic layer (`MoeaBench.metrics.evaluator`):

1. **`scale='raw'` (Volume Mode - Default Since v0.11.x)**: Returns the **Absolute Physical Volume** dominated by the solutions within the current Global Bounding Box. It does *not* divide the result by the best value. This stabilizes the inter-experiment values and ensures volumetric invariance. It answers: *"How much objective space has been physically conquered?"*
2. **`scale='ratio'` (Efficiency Mode)**: Computes the absolute volume and *divides* it by the maximum volume found in the provided experiments. This forces a `1.0` ceiling, showing exactly how close other algorithms got to the "State-of-the-Art" of that specific session. It answers: *"What is the competitive efficiency relative to session best?"*

By making `raw` the default, `MoeaBench` prioritizes scientific measurement and numerical stability over competitive ranking. If a competitive ranking is desired, the user must explicitly opt-in via `scale='ratio'`.

If true immutability is required across different plotting/analysis sessions, the user **must explicitly provide a fixed referencing point** (`nadir=[x, y, z]`).

## Consequences
- **Enhanced Integrity**: The library no longer conflates physical volume with relative competitive ranking by default. 
- **Numerical Stability**: Algorithm performance metrics no longer "shift" simply because a poorly performing neighbor was added to the Bounding Box.
- **Reporting Clarity**: The `MetricMatrix.report()` output dynamically frames the result as an answer to a core research question, improving interpretability.
