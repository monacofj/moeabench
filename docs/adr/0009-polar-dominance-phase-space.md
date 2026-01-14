# ADR 0009: Polar Dominance Phase Space

## Status
Proposed (2026-01-14)

## Context
Standard Multi-Objective Evolutionary Algorithm (MOEA) analysis usually focuses exclusively on the first Pareto front (Rank 1). While Population Stratification (ADR 0008) provides insights into the "onion layers" of the population, it doesn't quantify the quality of those layers relative to the front. We need a way to combine the **Structural Depth** (Rank) and **Convergence Quality** (Objective Norm) into a single diagnostic framework.

## Decision
We implement a **Polar Phase Space Analysis** that represents each dominance layer as a vector in a 2D coordinate system:

1.  **Coordinate Mapping**:
    - **X-axis**: Dominance Rank (1, 2, 3...).
    - **Y-axis**: Average Objective Norm ($\|F\|_2$) of individuals in that rank.

2.  **Polar Metrics**:
    - **Global Deficiency Index (GDI)**: The Euclidean magnitude ($\rho$) of the rank-quality vector. It measures the "Total Search Cost" of a layer.
    - **Population Maturity Index (PMI)**: The polar angle ($\theta = \arctan(Q/R)$). It measures the structural efficiency of the search.

3.  **Visualization (`mb.polarplot`)**:
    - Points are plotted in polar coordinates to create a "Fan" of search vectors.
    - The "lean" of the fan reveals the algorithm's state:
        - **Vertical Fan ($\theta \approx 90^\circ$)**: Immature/Elite-heavy. The algorithm has deep layers that are very far from the front.
        - **Shallow Fan ($\theta \to 0^\circ$)**: Mature/Compressed. Even deep layers are high-quality, indicating a unified convergence wave.

## Consequences

### Positive
- **Diagnosis of Search DNA**: Distinctly identifies different algorithm behaviors (e.g., "Elite Snipers" vs. "Collective Phalanxes").
- **Unified Metric**: GDI and PMI provide a single, sophisticated coefficient for population health that goes beyond binary Hypervolume.
- **Dimensionality Reduction**: Visualizes many-objective performance in a simple 2D polar view.

### Negative
- **Interpretation Curve**: Requires a basic understanding of polar coordinates to interpret maturity vs. deficiency.
- **Problem Dependency**: Early search on poorly-scaled problems may lead to extreme GDI values.
