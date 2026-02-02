# ADR 0016: Evolution of Scientific Validation and Guided Geometric Solvers

## Status
Accepted (Implemented in v0.7.6)

## Context
During the consolidation of v0.7.5 (ADR 0015), we identified that relying on "Heuristic Proxies" (Tier B) for constrained problems like DTLZ8 and uniform sampling in the decision space for biased problems like DTLZ4 introduced unacceptable geometric artifacts. Comparison with high-precision data from **Legacy2_optimal** revealed that "truth" based on stochastic searches suffered from convergence bias and spatial dispersion, compromising density-sensitive metrics (IGD and Hypervolume).

## Technical Decisions

### 1. Migration to Guided Analytical Solvers (DTLZ8)
We abandoned the "Union of Multiple Seeds" approach of NSGA-III for DTLZ8. 
*   **Decision**: We implemented a manifold reconstruction engine that inverts the problem's linear constraints. 
*   **Logic**: The solver samples the central curve and lateral "probes" of DTLZ8 deterministically, ensuring that every generated point strictly belongs to the intersection of active constraints. 
*   **Result**: DTLZ8 moves from Tier B (Probabilistic) to Tier A (Analytical), eliminating stochastic noise in the Ground Truth definition.

### 2. Angular Rectification of Biased Manifolds (DTLZ4)
DTLZ4 uses a power $x^{100}$ that collapses uniform samples of the decision space onto the axes of the sphere.
*   **Decision**: We replaced sampling in $X$ with uniform sampling in the angular space ($\Theta$).
*   **Logic**: The generator draws angles uniformly in the spherical triangle ($\Theta \in [0, \pi/2]$) and performs mathematical inversion ($x = \sqrt[100]{2\theta/\pi}$) to find decision parameters. 
*   **Consequence**: We ensure visually intact and statistically fair point density, eliminating the isolated "outliers" that distorted performance audits.

### 3. Legacy Benchmark Audit Protocol (VBar-Audit)
We established the audit against the `legacy2_optimal` as a scientific validation barrier.
*   **Decision**: Every change in a MOP's mathematical engine must be validated by a "Triple Benchmark" (v0.7.5 vs Legacy1 vs Legacy2) via `topo_shape`.
*   **Rigor**: Parity invariants ($\sum f = c$ or $\sum f^2 = c$) are verified in each audit to ensure that numerical precision maintains a residual error of less than $10^{-8}$.

## Narrative Interpretation and Consequences
This change marks the transition of MoeaBench from an "Optimization Framework" to a "Scientific Metrology Instrument". 

*   **Superior Rigor**: v0.7.6 now imposes a mathematical truth that is, in many cases, superior to data produced by state-of-the-art algorithms running for thousands of generations.
*   **Metric Stability**: Convergence metrics now operate on uniform density targets, reducing spurious variations in IGD and HV results caused by poor sampling of the optimal front.
*   **Auditability**: Replacing "frozen" CSV files with analytical solvers increases the framework's transparency, allowing any researcher to reproduce the ideal front without relying on random seeds or external data.

---
*Documented in 2026-01-25 to reflect the consolidation of the Z/Z2 Phase.*
