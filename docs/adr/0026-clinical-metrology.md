# ADR 0026: Clinical Metrology and Dimensional Calibration in Finite Sampling

**Status:** Proposed  
**Date:** 2026-02-06  
**Drivers:** Scientific Rigor, Scale Invariance, Fairness in Benchmarking  
**Affects:** `MoeaBench.diagnostics`, `MoeaBench.metrics`, Calibration Reports

---

## 1. Context and Problem Statement

### 1.1 The N=200 Audit Challenge
MoeaBench v0.8.0 introduced stringent diagnostic profiles (`RESEARCH`, `STANDARD`, `INDUSTRY`), originally defined by fixed thresholds based on the Pareto front diameter ($D$). For instance, the `STANDARD` profile required an IGD $< 0.5\% \times D$, and `RESEARCH` required $< 0.2\% \times D$.

During calibration with 3D problems (DTLZ1-4, DPF1-5) and a fixed population of $N=200$, a systemic anomaly was observed: visually excellent and converged algorithms were consistently failing as `SEARCH_FAILURE` in the stricter profiles (`STANDARD` and `RESEARCH`).

### 1.2 The Curse of Dimensionality in Coverage (The Sparsity Law)
Theoretical investigation revealed that fixed thresholds ignored the intrinsic geometry of the coverage problem.
To cover a manifold of $M$ objectives (with topological dimension $d \approx M-1$) using $N$ points, the expected average distance between neighbors ($R_{cov}$) does not tend to zero but rather to a physical limit imposed by density:

$$ R_{coverage} \propto N^{-\frac{1}{M-1}} $$

*   **In 2D ($M=2$):** With $N=200$, the expected distance is $\approx 0.5\%$. The fixed 0.5% threshold worked by coincidence.
*   **In 3D ($M=3$):** With $N=200$, the expected distance jumps to $\approx 7.0\%$. Requiring 0.5% is physically impossible, as it would demand more points than the population possesses.
*   **In Many-Ops ($M>3$):** Fixed requirements would lead to a 100% failure rate, regardless of algorithmic quality.

### 1.3 Refuting the "Perfect Ground Truth"
Furthermore, it was erroneously assumed that the Ground Truth (GT) is a perfect continuous surface. In practice, the GT is often a discrete set (2k to 10k points).
*   **Noise Floor:** Even if an algorithm selected the *best possible subset* of $N$ points from the GT, the IGD and EMD would not be zero. They would have an irreducible residual value ($IGD_{floor}$).
*   **Metrological Injustice:** Penalizing an algorithm for not reaching IGD=0.0 when the physical limit is IGD=0.03 is scientifically indefensible.

---

## 2. Architectural Decision: Separation of Physics and Clinic

To resolve this without compromising the reproducibility of canonical metrics, we adopt a **Separation of Concerns (SoC)** architecture inspired by medical practice.

### 2.1 Module `MoeaBench.metrics` (The Laboratory)
*   **Responsibility:** Calculate pure physical quantities.
*   **Nature:** Absolute, "Dumb", Reproducible.
*   **Functions:** `igd(P, GT)`, `emd(P, GT)`.
*   **Decision:** These functions remain unchanged. We will not embed "magic compensations" here. This ensures MoeaBench's IGD is comparable to literature.

### 2.2 Module `MoeaBench.clinic` (The Clinic)
*   **Responsibility:** Interpret physical quantities in light of context (Budget $N$, Dimensionality $M$).
*   **Nature:** Relative, Calibrated, Contextual.
*   **Key Concept:** **Efficiency Ratio to Floor ($Efficiency Ratio$)**.
    
    $$ Metric_{eff} = \frac{Metric_{observed}}{Metric_{floor}(N, Problem)} $$
    
*   **Interpretation:**
    *   $1.0$: Perfect Efficiency (Algorithm reached the physical sampling limit).
    *   $1.2$: 20% Inefficiency over the physical limit.
    *   $>2.0$: Significant degradation.

### 2.3 Module `MoeaBench.diagnostics` (The Auditor)
*   **Responsibility:** Issue verdicts (`PASS`/`FAIL`) based on clinical data.
*   **Decision:** Profiles (`RESEARCH`, `STANDARD`) are no longer defined by arbitrary physical values but by **Efficiency Levels**.
    *   **Research:** Requires Efficiency $\le 1.10$ (10% tolerance over optimal).
    *   **Standard:** Requires Efficiency $\le 1.50$ (50% tolerance).
*   **Benefit:** This definition is **Scale Invariant**. It works equally for 2D, 3D, or 10D, as the dimensional difficulty has been absorbed by the denominator ($Metric_{floor}$).

---

## 3. Technical Specification

### 3.1 Statistical Baseline Generation (`scripts/generate_baselines.py`)
We will not use approximate analytical formulas. We will use **Monte Carlo Methods** to determine the empirical floor for each problem.

1.  **Sampling:** For each problem, we extract $K=100$ random subsets of size $N$ from the GT.
2.  **Floor Calculation:**
    *   **IGD Floor ($P_{10}$):** The 10th percentile of the IGD metrics of the subsets. We use $P_{10}$ ("Best-of-K") instead of the median ($P_{50}$) to represent a target of "Algorithmic Excellence", not just "Typical Randomness".
    *   **EMD Floor ($P_{10}$):** Calculated between **pairs of subsets** ($S_a, S_b$) to isolate discretization noise and avoid cardinality artifacts ($N$ vs Full GT).

The result is persisted in `MoeaBench/diagnostics/resources/baselines.json`.

### 3.2 New Clinical Metrics (`MoeaBench/clinic/indicators.py`)

#### `igd_efficiency(P, GT, n_eval, ...)`
1.  Loads the `igd_floor` corresponding to the problem and $N$.
2.  Calculates `raw_igd = metrics.igd(P, GT)`.
3.  Returns `raw_igd / igd_floor`.

#### `purity_robust(P, GT, percentile=95)`
1.  Calculates GD distances for all points ($d_i$).
2.  Returns the 95th percentile of $d_i$.
3.  **Justification:** Mean GD is unstable. A single outlier point can destroy the mean. P95 trims the edges and focuses on the convergence of the population's main mass, aligning better with the visual diagnosis of `NOISY_POPULATION`.

---

## 4. Consequences and Advantages

### 4.1 Metrological Fairness
Algorithms will no longer be punished for the limitations of physics. A `FAIL` diagnosis will unequivocally mean "worse than possible", not "worse than the auditor's imagination".

### 4.2 Scalability to Many-Objectives
The system is ready for $M=4, 5, 10...$. As dimensionality grows, `IGD_floor` grows organically. The pass criterion (Ratio $\le 1.5$) remains stable and meaningful.

### 4.3 Semantic Clarity
*   **Search Failure:** Now reserved for actual convergence failures (Ratio $> 3.0$ or similar).
*   **Noisy Population:** Diagnosed via `purity_robust`, indicating local dispersion without global collapse.

This architecture moves MoeaBench from a "Plotting" tool to a **Precision Scientific Audit** platform, capable of distinguishing fundamental limitations from algorithmic failures.
