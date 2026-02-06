# ADR 0025: Comprehensive Algorithmic Pathology (The Truth Table)

## Context
In multi-objective optimization, "performance" is a complex amalgam of convergence (proximity to the true front), coverage (finding the full extent of the front), and uniformity (even spacing of solutions). Traditional analysis often treats metrics like Generational Distance (GD), Inverted Generational Distance (IGD), and Earth Mover's Distance (EMD) in isolation. However, it is the *interaction* between these metrics that reveals the true behavior—or pathology—of an algorithm.

We identified three primary binary states for each metric: **Good (B)** vs. **Bad (R)**. This creates a combinatorial truth table of $2^3 = 8$ possible states.

## Decision
We will implement a comprehensive diagnostic system that maps *every* possible combination of these three metrics to a specific, scientifically grounded diagnosis. This moves the framework from simple "pass/fail" reporting to a nuanced expert system capable of identifying subtle behaviors like "Distribution Bias" or "Shadow Fronts."

## The Certification Framework (v0.9)

In v0.9, we transition from absolute values to scale-invariant metrics normalized by the **Characteristic Diameter ($D$)** of the reference front. $D$ is defined as the bounding box diagonal of the Ground Truth.

Metrics are normalized as: $nMetric = Metric_{absolute} / D$.

### Quality Profiles (Precision Tiers)
Thresholds are defined as fractions of $D$.

| Profile | nGD (p) | nIGD (2p) | nEMD (3p) | Clinical Utility |
| :--- | :---: | :---: | :---: | :--- |
| **EXPLORATORY** | 10.0% | 20.0% | 30.0% | "Legacy v0.8.0 safety net / Early exploration." |
| **INDUSTRY** | 1.0% | 2.0% | 3.0% | "Good enough for decision making." |
| **STANDARD** | 0.5% | 1.0% | 1.5% | "Standard benchmarking quality." |
| **RESEARCH** | 0.2% | 0.5% | 0.8% | "State-of-the-art / Publication quality." |

## The Pathology Truth Table

The logical interaction remains constant across all profiles. **L** = Low/Good (below threshold), **H** = High/Bad (above threshold).

| nGD | nIGD | nEMD | Diagnosis | Mathematical Signature |
| :---: | :---: | :---: | :--- | :--- |
| **L** | **L** | **L** | **IDEAL_FRONT** | Hit the target, covered it, correct density. |
| **L** | **L** | **H** | **BIASED_SPREAD** | Converged & Covered, but clustered/distorted. |
| **L** | **H** | **L** | **GAPPED_COVERAGE** | Local correctness, but global holes (Swiss Cheese). |
| **L** | **H** | **H** | **COLLAPSED_FRONT** | Degeneracy. Collapsed to a point/line. |
| **H** | **L** | **L** | **NOISY_POPULATION** | Good core coverage, but many dominated points remain. |
| **H** | **L** | **H** | **DISTORTED_COVERAGE** | Covers the area, but with noise and wrong shape. |
| **H** | **H** | **L** | **SHIFTED_FRONT** | Right shape, wrong location (Local Optimum Trap). |
| **H** | **H** | **H** | **SEARCH_FAILURE** | Complete failure to locate or cover the front. |

### The Statistical Floor (Future)
When the population size $N$ is small, $nIGD$ and $nEMD$ have a non-zero lower bound due to sampling noise. Future versions will adjust thresholds dynamically if the theoretical "Sampling Noise Floor" is higher than the profile threshold.

## Consequences
*   **Positive**: Eliminates confusing "Contradiction" messages.
*   **Positive**: Distinguishes between "Shadow Fronts" (Systematic Shift) and "Noisy Fronts" (Stochastic Variance).
*   **Negative**: Computationally expensive (EMD is $O(N^3)$).
