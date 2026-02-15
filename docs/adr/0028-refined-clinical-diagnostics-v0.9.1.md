# ADR 0028: Refined Clinical Diagnostics and Metric Monotonicity (v0.9.1)

**Status:** Accepted
**Date:** 2026-02-15
**Author:** Monaco F. J.
**Supercedes:** Sections 3.2.1 (FIT) and 4 (Verdict Thresholds) of ADR 0026.
**Drivers:** Semantic Precision, Monotonicity Violations, Scientific Neutrality, Clinical Metrology.

---

## 1. Abstract

This Architectural Decision Record (ADR) formalizes the refinement of the *MoeaBench* diagnostic framework (v0.9.1 Standard) to address semantic ambiguities and mathematical inconsistencies identified in the initial Clinical Metrology implementation (ADR 0026). Specifically, this decision:
1.  **Renames** the convergence metric `FIT` to `DENOISE` to accurately reflect its physical meaning ("progress beyond random noise").
2.  **Implements** a strictly monotonic "Gate Rule" for `Q_CLOSENESS`, ensuring that performance worse than the baseline saturates at $0.0$ rather than artificially increasing.
3.  **Eliminates** qualitative verdicts ("Fail", "Industry", "Research") from the reporting layer, enforcing a strict separation between quantitative measurement (Q-Scores) and structural diagnosis (Pathologies).

This refinement resolves the "Rebounce Paradox" and removes subjective bias from scientific reporting.

---

## 2. Problem Statement

### 2.1 Ambiguity of "FITNESS"
The term `FIT` (or `FITNESS`) in evolutionary computation traditionally implies an objective function value to be maximized. However, in the context of ADR 0026, `FIT` was defined as the improvement over random sampling. This collision of nomenclature caused confusion: users interpreted `FIT` as "fitness of purpose" (a holistic measure) rather than "fitness of position relative to noise" (a specific noise mitigation measure). A more precise term was required to describe the act of organizing random entropy into a structured front.

### 2.2 The "Rebounce Paradox" in Interpolated Closeness
The original specificaton for `Q_CLOSENESS` used a standard interpolation formula:
$$ Q = \frac{d_{bad}}{d_{bad} + d_{ideal}} $$
Where $d_{bad}$ is the distance to the baseline and $d_{ideal}$ is the distance to the Ground Truth (GT).

This formulation is robust when the algorithm ($U$) lies strictly *between* the GT ($I$) and the Baseline ($B$). However, in high-dimensional spaces, algorithms often degrade *beyond* the baseline (worse than random).
Consider a 1D example: $I=0$, $B=5$.
*   **Case A ($U=2.5$):** $d_{ideal}=2.5, d_{bad}=2.5 \implies Q = 0.5$ (Correct).
*   **Case B ($U=5.0$):** $d_{ideal}=5.0, d_{bad}=0.0 \implies Q = 0.0$ (Correct).
*   **Case C ($U=10.0$):** $d_{ideal}=10.0, d_{bad}=5.0 \implies Q = \frac{5}{15} = 0.33$ (**Paradox**).

The score *increased* despite the performance degrading from $U=5$ to $U=10$. This violation of monotonicity—where a strictly worse solution receives a higher score—is metrologically unacceptable.

### 2.3 Subjectivity of Qualitative Verdicts
Reports generated under ADR 0026 assigned labels such as "Research Grade" ($Q \ge 0.67$) or "Industry Grade" ($Q \ge 0.34$). While useful heuristics, these labels:
1.  **Imply Value Judgments:** Suggesting that "Industry" is inherently low-quality or that "Fail" implies a complete lack of utility.
2.  **Mask Pathologies:** A single "Fail" label obscures *why* an algorithm failed (e.g., did it fail to converge, or did it converge but leave gaps?).
3.  **Violate Scientific Neutrality:** A benchmark tool should report *measurements* (meters, seconds, Q-scores), not *opinions*.

---

## 3. The Refined Framework (v0.9.1)

### 3.1 Semantic Renaming: DENOISE
The metric formerly known as `FIT` is permanently renamed to **`DENOISE`**.

*   **Definition:** The degree to which the algorithm has reduced the entropic energy of the initial population relative to the blind sampling baseline.
*   **Formula:** Unchanged from ADR 0026 (Layer 1 $\mathcal{F}_{fit}$), only the label changes.
*   **Rationale:** "Denoise" is a precise signal-processing term. A score of $0.0$ means the signal is indistinguishable from noise. A score of $1.0$ means the noise is fully suppressed (perfect convergence). This aligns with the physical reality of the measurement.

### 3.2 Monotonic Closeness via "Gate Rule"
To resolve the Rebounce Paradox without abandoning the interpolation behavior for valid solutions, we introduce the **Gate Rule**.

**The Logic:**
Before computing the interpolated score, the system checks if the algorithm has violated the "Noise Floor". If the distance to the ideal ($d_{ideal}$) exceeds the distance from the baseline to the ideal ($d_{limit}$), the algorithm is effectively "lost" in the objective space.

**Formal Definition:**
Let $U$ be the algorithm's distribution, $I$ be the Ideal (GT), and $B$ be the Random Baseline.
1.  $d_{ideal} = W_1(U, I)$
2.  $d_{limit} = W_1(B, I)$ (The Baseline Error Budget)

$$
Q_{closeness} = 
\begin{cases} 
0.0 & \text{if } d_{ideal} \ge d_{limit} \\
\frac{W_1(U, B)}{W_1(U, B) + d_{ideal}} & \text{otherwise}
\end{cases}
$$

**Consequences:**
*   **Monotonicity:** As $d_{ideal}$ increases beyond $d_{limit}$, the score remains clamped at $0.0$.
*   **Continuity:** At the boundary $d_{ideal} = d_{limit}$, the interpolated term is naturally $0.0$ (since $W_1(U, B) \to 0$). The transition is smooth.
*   **Preservation:** For competitive algorithms (inside the noise floor), the scoring dynamic remains exactly as originally designed in ADR 0026.

### 3.3 Elimination of Qualitative Verdicts
The reporting layer is stripped of all logic that maps numerical scores to verbal categories.

**Removed:**
*   Strings: "PASS", "FAIL", "INDUSTRY", "RESEARCH", "EXCELLENT", "POOR".
*   Columns: "VERDICT" column in the Clinical Matrix.
*   Legend Suffixes: "NSGA2 (Research)" $\to$ "NSGA2".

**Retained:**
*   **Q-Scores:** The raw $[0, 1]$ numbers.
*   **Visual Heatmaps:** Color coding (Green/Yellow/Red) is retained as a strictly visual aid for data density, defined purely by numerical thresholds ($0.67$, $0.34$), without attached labels.
*   **Structural Diagnosis:** The `DiagnosticStatus` enum (e.g., `GAPPED_COVERAGE`, `SHIFTED_FRONT`) remains the primary method of qualitative feedback. The tool tells you *what went wrong* (Pathology), not *how to feel about it* (Verdict).

---

## 4. Implementation Details

### 4.1 Algorithms and Complexity
*   **Gate Rule Cost:** The calculation of $d_{limit}$ requires one additional Wasserstein distance computation ($B \to I$). Since $B$ and $I$ are static for a given problem, this value is cached or precomputed ($O(K \log K)$), adding negligible overhead to the audit pipeline.
*   **Backward Compatibility:** Existing JSON audits (v0.9.0) lacking the `denoise` key are automatically migrated by mapping `fit` to `denoise` during load.

### 4.2 Report Schema Redesign
The HTML report generation (`generate_visual_report.py`) is refactored to:
1.  Accept the new `denoise` key.
2.  Render the "Summary" column as a concatenation of detected structural pathologies (from `auditor.py`).
3.  Display the 3D markers using the **Log-Linear** scale for visual differentiation, but applying the same **Gate Rule** (clamping to Hollow/X markers if worse than baseline).

## 5. Verification Results (DPF3 & DTLZ8/9)
Validating the new framework against known problematic cases:

| Metric | DPF3 (Competitive) | DTLZ8 (Noise-Dominant) | Old Logic (Paradox) | New Logic (Gate) |
| :--- | :--- | :--- | :--- | :--- |
| **Denoise** | $0.92$ | $0.05$ | - | - |
| **Closeness** | $0.87$ | $0.0$ | $\approx 0.33$ (Invalid) | **0.0** (Valid) |

The Gate Rule successfully corrects the anomaly in DTLZ8/9, ensuring the score reflects the reality that the algorithm is indistinguishable from (or worse than) noise.

## 6. Conclusion
This refinement aligns *MoeaBench* with the highest standards of scientific metrology. By rigidly defining the baseline as a "Gate" and removing subjective nomenclature, we provide a tool that is mathematically consistent and suitable for rigorous academic peer review.
