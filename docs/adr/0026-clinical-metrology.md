# ADR 0026: Clinical Metrology and Dimensional Calibration in Finite Sampling Regimes

**Status:** Accepted
**Date:** 2026-02-09
**Author:** Monaco F. J.
**Drivers:** Scientific Rigor, Scale Invariance, Algorithmic Auditability, Metrological Fairness

---

## 1. Abstract

This Architectural Decision Record (ADR) formalizes the transition of the *MoeaBench* diagnostic system from a threshold-based heuristic model to a **Clinical Metrology** architecture. The central innovation is the decoupling of physical measurements ("Fair Metrics") from their engineering interpretation ("Q-Scores"). We demonstrate that finite sampling regimes ($N \approx 100-200$) in high-dimensional manifolds ($M \ge 3$) impose intrinsic geometric limitations ("The Sparsity Law") that render fixed-threshold metrics scientifically unrealistically exigent. To address this, we introduce a **3-Layer Architecture** (Raw $\to$ Fair $\to$ Q-Score) anchored by rigorous statistical baselines. This system provides a scale-invariant, audit-proof framework for certifying algorithmic performance, distinguishing fundamental physical bounds from actual search failures.

---

## 2. Problem Statement: The Crisis of Naive Metrology

### 2.1 The Curse of Dimensionality in Finite Sampling
Traditional auditing profiles (e.g., standard industrial thresholds of $IGD < 0.01$) rely on the implicit assumption that the *Pareto Front* is a continuous surface that can be densely approximated by the algorithmic population $P$.
However, in practical engineering scenarios, the population size $N$ is physically constrained (typically $N \le 200$). As the number of objectives $M$ increases, the theoretical minimum distance required to "cover" the manifold grows exponentially with respect to the dimensionality $d = M-1$.

We define the **Sparsity Law** for a $d$-dimensional manifold covered by $N$ points:
$$ R_{cov} \propto N^{-\frac{1}{d}} $$

*   **For $M=2$ ($d=1$):** With $N=100$, coverage density is high ($R \approx 0.01$). Fixed thresholds work.
*   **For $M=3$ ($d=2$):** With $N=100$, the expected nearest-neighbor distance jumps to $R \approx 0.10$. A threshold of $0.01$ becomes physically impossible, regardless of algorithmic perfection.
*   **For $M>3$:** The manifold becomes largely empty space. Comparing an algorithm against a "Perfect Zero" baseline measures the geometry of the problem, not the quality of the solver.

### 2.2 The Fallacy of the Perfect Ground Truth
Furthermore, in numerical benchmarking, the "Ground Truth" (GT) is often a discrete reference set $F_{opt}$, not an infinite surface. Even an **Optimal Subset** $S_{opt} \subset F_{opt}$ of size $N$ will exhibit non-zero IGD and EMD values due to discretization noise.
Evaluating an algorithm against $IGD=0.0$ imposes an unfair penalty, effectively failing robust algorithms for not achieving a physically attainable state—a metrological injustice.

---

## 3. Theoretical Framework: The 3-Layer Architecture

To resolve these inconsistencies, we adopt a **Clinical Metrology** approach, separating the physics of measurement from the judgment of quality.

### 3.1 Layer 0: Raw Metrics (`MoeaBench.metrics`)
*   **Definition:** Pure mathematical distances in the Euclidean objective space ($\mathbb{R}^M$).
*   **Role:** The absolute "truth" of position. Uncorrected and unnormalized.
*   **Examples:** $GD(P, GT)$, $IGD(P, GT)$, $W_1(P, GT)$.
*   **Status:** Immutable. These are the primitives upon which all higher logic is built.

### 3.2 Layer 1: Fair Metrics (`MoeaBench.diagnostics.fair`)
*   **Definition:** Physically meaningful quantities, corrected for scale and resolution artifacts.
*   **Direction:** **Low-is-Better** (Physical Error).
*   **Role:** To characterize specific "pathologies" (e.g., Non-convergence, Gaps, Irregularity) in a way that is comparable across different problems.
*   **Formal Definitions:**
    1.  **FAIR_FIT (Convergence):**
        $$ \mathcal{F}_{fit} = \frac{GD_{95}(P \to GT)}{s_{GT}} $$
        *Normalization by $s_{GT}$ (GT resolution) makes convergence dimensionless.*
    2.  **FAIR_COVERAGE (Completeness):**
        $$ \mathcal{F}_{cov} = IGD_{mean}(GT \to P) $$
    3.  **FAIR_GAP (Continuity):**
        $$ \mathcal{F}_{gap} = IGD_{95}(GT \to P) $$
        *Captures the "largest hole" in coverage, robust to outliers.*
    4.  **FAIR_REGULARITY (Geometry):**
        $$ \mathcal{F}_{reg} = W_1(\text{NN}(P), \text{NN}(U_{ref})) $$
        *Wasserstein distance between the Nearest-Neighbor distribution of $P$ and a reference Uniform Lattice $U_{ref}$.*
    5.  **FAIR_BALANCE (Topology):**
        $$ \mathcal{F}_{bal} = D_{JS}(\text{Hist}(P) || \text{Hist}(U_{ref})) $$
        *Jensen-Shannon divergence of cluster occupancy.*

### 3.3 Layer 2: Q-Scores (`MoeaBench.diagnostics.qscore`)
*   **Definition:** Engineering quality grades normalized to the expectation of the specific problem instance $(Problem, N, M)$.
*   **Direction:** **High-is-Better** ($Q \in [0.0, 1.0]$).
*   **Role:** To determine the "Verdict" (Pass/Fail) by Contextualizing the Fair Metric.
*   **The Transformation:**
    $$ Q = 1.0 - \text{clip}\left( \frac{\mathcal{F}_{observed} - \mathcal{F}_{ideal}}{\mathcal{F}_{random} - \mathcal{F}_{ideal}} \right) $$

    *   $\mathcal{F}_{ideal}$: The expected Fair Metric value of an **Optimal Subset** (obtained via Farthest Point Sampling of GT).
    *   $\mathcal{F}_{random}$: The expected value of a **Random Subset** (or BBox Random).

---

## 4. Implementation Specification

### 4.1 Strict Baseline Policy (The "Fail-Closed" Principle)
To ensure the integrity of the audit, the system must never "guess" or "fallback".
*   **Fail-Closed:** If the baselines ($\mathcal{F}_{ideal}, \mathcal{F}_{random}$) for a specific triplet $(Problem, K, Metric)$ are missing from the certified JSON, the system raises `UndefinedBaselineError`. The audit is aborted. This prevents "Shadow Audits" where algorithms pass due to loose default thresholds.

### 4.2 Baseline Selection Rules
To prevent grade inflation, the baselines are generated with rigorous statistical methods:
*   **FIT Exception:** For Convergence, a random subset of GT is *too easy*. A "Random" algorithm would be far worse (generating points anywhere in the BBox).
    *   $\mathcal{F}_{ideal}^{fit} = 0.0$ (Physical perfection is required).
    *   $\mathcal{F}_{random}^{fit} = \mathbb{E}[\text{BBox Sampling}]$.
*   **Diversity Metrics (Cov, Gap, Reg, Bal):**
    *   $\mathcal{F}_{ideal} = \text{Median}(FPS(GT, K))$
    *   $\mathcal{F}_{random} = \text{Median}(RandSubset(GT, K))$

### 4.3 Modular Architecture (`MoeaBench.diagnostics`)
The architecture is reified in a clean Python package:
*   `fair.py`: Pure physics engines (scipy-based). No domain logic.
*   `qscore.py`: The "Judge". Holds the normalization logic and baseline connectivity.
*   `baselines.py`: The "Vault". secure, cached access to the baseline V2 JSON.
*   `auditor.py`: The "Orchestrator". Computes the 5-dimensional quality matrix and issues the final textual verdict.

---

## 5. Implications and Review

### 5.1 Audit Traceability
Every cell in the final report acts as a verifiable proof. The tooltips expose the entire derivation chain:
> **Q-Score: 0.85**
> *("Good", $Q > 0.67$)*
>
> **Trace:**
> *   **Fair Metric:** 0.045 (Physical measurement)
> *   **Ideal (Reference):** 0.012 (Limit of physics for N=100)
> *   **Random (Noise):** 0.150 (Expected chaos)
> *   **Sample Size:** K=100 (Raw=100)

This transparency silences the "Why did I fail?" question. The user can see instantly if they failed because their physics were bad ($\approx Random$) or simply not good enough for the strict context ($\approx Ideal$).

### 5.2 Scale Invariance Achieved
By normalizing against the *Ideal* and *Random* specific to that dimensionality, the Q-Score becomes a **Universal Quality Constant**. A $Q=0.9$ on a 2D problem implies the same level of mastery as a $Q=0.9$ on a 10D problem, allowing for cross-problem aggregation and ranking.

### 5.3 Conclusion
MoeaBench v0.9 moves beyond simple plotting. It establishes a **Metrological Standard** for multi-objective auditing. By acknowledging the sparsity law and certifying against rigorous empirical baselines, we provide the first truly fair and scalable framework for algorithmic benchmarking in the many-objective era.
