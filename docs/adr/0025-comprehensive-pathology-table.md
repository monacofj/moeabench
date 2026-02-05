# ADR 0025: Comprehensive Algorithmic Pathology (The Truth Table)

## Context
In multi-objective optimization, "performance" is a complex amalgam of convergence (proximity to the true front), coverage (finding the full extent of the front), and uniformity (even spacing of solutions). Traditional analysis often treats metrics like Generational Distance (GD), Inverted Generational Distance (IGD), and Earth Mover's Distance (EMD) in isolation. However, it is the *interaction* between these metrics that reveals the true behavior—or pathology—of an algorithm.

We identified three primary binary states for each metric: **Good (B)** vs. **Bad (R)**. This creates a combinatorial truth table of $2^3 = 8$ possible states.

## Decision
We will implement a comprehensive diagnostic system that maps *every* possible combination of these three metrics to a specific, scientifically grounded diagnosis. This moves the framework from simple "pass/fail" reporting to a nuanced expert system capable of identifying subtle behaviors like "Distribution Bias" or "Shadow Fronts."

## The Pathology Truth Table (v0.8.0)

We consider the following metrics:
1.  **Convergence (GD)**: Is the population close to the true Pareto Front?
2.  **Coverage (IGD)**: Does the population represent the entire Pareto Front?
3.  **Topology (EMD)**: Does the shape/distribution match the Ground Truth?

**Legend**: L = Low/Good, H = High/Bad.

| GD | IGD | EMD | Diagnosis | Mathematical Signature |
| :---: | :---: | :---: | :--- | :--- |
| **L** | **L** | **L** | **IDEAL_FRONT** | Hit the target, covered it, correct density. |
| **L** | **L** | **H** | **BIASED_SPREAD** | Converged & Covered, but clustered/distorted. |
| **L** | **H** | **L** | **GAPPED_COVERAGE** | Local correctness, but global holes (Swiss Cheese). |
| **L** | **H** | **H** | **COLLAPSED_FRONT** | Degeneracy. Collapsed to a point/line. |
| **H** | **L** | **L** | **NOISY_POPULATION** | Good core coverage, but low purity (many dominated points). |
| **H** | **L** | **H** | **DISTORTED_COVERAGE** | Covers the area, but with noise and wrong shape. |
| **H** | **H** | **L** | **SHIFTED_FRONT** | Right shape, wrong location (Local Optimum Trap). |
| **H** | **H** | **H** | **SEARCH_FAILURE** | Complete failure to locate or cover the front. |

## Implementation Details

### Thresholding
To convert continuous metrics into binary "Good/Bad" states, we use adaptive thresholds calibrated for engineering practicality:
*   **Good GD**: $< 0.1$ (Relaxed from 1e-3 for robust acceptance)
*   **Good IGD**: $< 0.1$ (Standard coverage)
*   **Good EMD**: $< 0.12$ (Visual tolerance threshold)

### The "Impossible" States Resolution
Previous versions considered (High GD, Low IGD) as a contradiction. In v0.8.0, this is correctly reclassified as **Noisy Population**: the algorithm *did* cover the front (hence Low IGD), but the presence of many outliers inflates the average distance (High GD).

## Consequences
*   **Positive**: Eliminates confusing "Contradiction" messages.
*   **Positive**: Distinguishes between "Shadow Fronts" (Systematic Shift) and "Noisy Fronts" (Stochastic Variance).
*   **Negative**: Computationally expensive (EMD is $O(N^3)$).
