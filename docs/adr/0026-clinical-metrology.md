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
        $$ \mathcal{F}_{fit} = \frac{GD_{95}(P \to GT)}{s_K} $$
        *  $s_K = \text{Median}(\text{NN}(U_K))$, where $U_K = \text{FPS}(GT, K, \text{seed}=0)$.
        *  **Rationale for Sanitization:**
            *   **The "Microscopic Ruler" Problem:** The previous normalizer, $s_{GT}$, is the resolution of the dense Ground Truth ($N \approx 10,000$). In high dimensions ($M \ge 3$), this value becomes infinitesimal ($10^{-5}$), causing raw distances ($10^{-1}$) to explode when normalized.
            *   **Dynamic Range Collapse:** Because the Random Baseline is also measured against this microscopic ruler, its value becomes astronomical (e.g., ~2400 for DTLZ6). The Q-Score formula ($1 - \text{Error}/\text{Random}$) then divides a large error by a colossal baseline, compressing all scores to $0.99+$ even for failed runs.
            *   **The Clinical Fix ($s_K$):** We replace the microscopic ruler with a macroscopic one: the expected resolution of a finite population of size $K$. This restores the dynamic range, ensuring that a gross visual error penalizes the Q-Score effectively.
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
*   **Definition:** Engineering quality grades normalized to the expectation of the specific problem instance $(Problem, N, M)$ using Empirical Distribution Functions.
*   **Direction:** **High-is-Better** ($Q \in [0.0, 1.0]$).
*   **Role:** To determine the "Verdict" (Pass/Fail) by Contextualizing the Fair Metric within the actual population of random outcomes.
*   **The Transformation (ECDF-based):**
    To eliminate distortions caused by skewed or fat-tailed baseline distributions, we replace linear interpolation with a logic based on the **Empirical Cumulative Distribution Function (ECDF)**, denoted as $F_{rand}(x)$:

    $$ Q = 1.0 - \text{clip}\left( \frac{F_{rand}(\mathcal{F}_{observed}) - F_{rand}(\mathcal{F}_{ideal})}{F_{rand}(\mathcal{F}_{random}) - F_{rand}(\mathcal{F}_{ideal})} \right) $$

    *   $F_{rand}(x)$: The ECDF of a certified random baseline distribution (200 samples).
    *   $\mathcal{F}_{ideal}$: The expected Fair Metric value of an **Optimal Subset**.
    *   $\mathcal{F}_{random}$: The median of the baseline distribution (guaranteed $Q=0$ at the median).

---

## 4. Implementation Specification

### 4.1 Strict Baseline Policy (Baselines v4)
To ensure the integrity of the audit, the system must never "guess" or "fallback".
*   **Baselines v4 (ECDF):** The system stores a discrete distribution of **200 sorted samples** per triplet $(Problem, K, Metric)$.
*   **Fail-Closed:** If the baselines for a specific triplet are missing or do not conform to the 200-sample sorted schema, the system raises `UndefinedBaselineError`.
*   **Consistency Invariant:** The stored `rand50` MUST be the median of the 200 samples in `rand_ecdf`. The system verifies this at load time.

### 4.2 Baseline Selection Rules
To prevent grade inflation, the baselines are generated with rigorous statistical methods:
*   **FIT Exception:** For Convergence, a random subset of GT is *too easy*. A "Random" algorithm would be far worse (generating points anywhere in the BBox).
    *   $\mathcal{F}_{ideal}^{fit} = 0.0$ (Physical perfection is required).
    *   $\mathcal{F}_{random}^{fit} = \mathbb{E}[\text{BBox Sampling} / s_K]$.
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

## 6. Structural Evidence: The Role of Distance-to-GT CDF

While the **Clinical Matrix** (Layer 2) provides the engineering verdict, the **Distance-to-GT CDF** (Cumulative Distribution Function) serves as the **Structural Evidence** layer. It bridges the gap between raw mathematical distances and clinical interpretation.

### 6.1 Diagnostic Anatomy of the CDF Curve
The shape of the CDF curve allows for the "physical biopsy" of algorithmic failure:

1.  **Steep Curve (Left-aligned):** High-precision convergence. The population is uniformly close to the manifold. (Healthy Profile).
2.  **Long Tail (Right-skewed):** Presence of significant outliers or solutions trapped in local optima. Explains poor **REG (Regularity)** scores.
3.  **Rigid Shift:** The curve shape is correct but moved right on the X-axis. Indicates the algorithm found the correct geometry but failed the final convergence step. Explains low **FIT** scores.
4.  **Discontinuous Plateaus:** Vertical gaps in the CDF indicate "empty" regions in the Pareto approxmiation. Direct evidence for high **GAP** indices.

### 6.2 Conclusion
By institutionalizing the CDF as a mandatory validation view, *MoeaBench* ensures that every quantitative Q-Score is accompanied by a transparent, visually inspectable proof of the underlying search pathology.

---

## 7. Implications and Review
