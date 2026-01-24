# ADR 0015: Scientific Ground Truth and Numerical Integrity

## Status
Proposed

## Context
MoeaBench requires a definitive "Ground Truth" for auditing multi-objective optimization algorithms. Numerical floating-point arithmetic (IEEE 754) introduces non-deterministic noise across CPU architectures (Intel vs ARM) and linear algebra backends (MKL, OpenBLAS). To mitigate this and ensure scientific reproducibility, we establish a rigorous "Tabula Rasa" ground truth protocol.

## Technical Decision

### 1. Hybrid Ground Truth Architecture
We categorize MOPs into two certification tiers based on the availability of analytical samplers:

#### Tier A: Analytical Ground Truth (Deterministic)
Applied to: **DTLZ1-7, DTLZ9, DPF1-5**.
*   **Generation Method**: Direct coordinate sampling via `MOP.ps(n_points)` followed by `MOP.evaluation(X)`.
*   **Sampling Density**: $N_{points} = 500$ per front.
*   **Scale Invariance**: Variable count $N$ is dynamically set as $N = \max(M + 20, 31)$ to ensure that distance functions ($g$) are evaluated beyond minimal default configurations, testing the robustness of vectorized implementations.
*   **Degeneracy (DPF)**: Reference dimension $D$ is fixed at $2$ to maintain extreme manifold degeneracy in high objective spaces ($M \in \{3, 5, 10, 31\}$).

#### Tier B: Heuristic Ground Truth Proxy (Probabilistic)
Applied to: **DTLZ8** (Constraint-dominated problem lacking a closed-form analytical sampler).
*   **Algorithm**: NSGA-III (Reference-point based) for high-dimensional coverage.
*   **Population Size**: $200$ for $M < 10$; $400$ for $M \ge 10$.
*   **Search Depth**: $500$ generations per run.
*   **Aggregation Strategy**: High-Fidelity Multi-Seed Union. 
    *   Executed over 5 independent seeds: $\{1, 2, 3, 4, 5\}$.
    *   Resulting fronts ($F_{seed}$) are vertically stacked ($F_{pool} \in \mathbb{R}^{5PN \times M}$).
    *   Final reference set is the non-dominated subset of $F_{pool}$, ensuring maximum boundary discovery.

### 2. Numerical Integrity Protocol (IEEE 754 Robustness)
Verification of the "Analytical Brain" and MOEA results against the ground truth must follow these tolerances:

*   **Absolute Tolerance (`atol`)**: $1 \times 10^{-12}$ (Handling underflow and zero-comparisons).
*   **Relative Tolerance (`rtol`)**: $1 \times 10^{-8}$ (Handling scaling across objectives).
*   **Global Seed Isolation**: All generation scripts must anchor `numpy.random.seed(42)` and MOEA-specific seeds to prevent stochastic drift in the reference CSVs.

### 3. Verification of Mathematical Invariants
Integrity is defined by geometric invariants rather than coordinate identity:
*   **Linear (DTLZ1)**: $\sum_{i=1}^M f_i = 0.5$
*   **Spherical (DTLZ2-6, DPF1,2,5)**: $\sum_{i=1}^M f_i^2 = 1.0$
*   **Disconnected (DTLZ7)**: $f_M = 2M - \sum_{i=1}^{M-1} f_i (1 + \sin(3 \pi f_i))$ (for optimal $g=1$).
*   **Curve (DTLZ9)**: Boundary condition $f_1 = f_2 = \dots = f_{M-1}$ AND constraint $\sum_{j=1}^{M-1} (f_M^2 + f_j^2) = 1.0$.

## Consequences
*   **Version Pinning**: Ground Truth CSVs are artifacts tied to the library's state at v0.7.5 but hosted in a "Tabula Rasa" structure (`tests/ground_truth/`) where the Git history serves as the evolution ledger.
*   **Platform Portability**: Tests in CI environments (GitHub Actions) and local dev machines must yield identical pass/fail veridicts regardless of BLAS/LAPACK optimizations.
*   **Auditor Independence**: The generation script (`generate_truth.py`) provides a transparent, auditable process for any third-party scientific review.
