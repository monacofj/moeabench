<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->
# Clinical Quality Matrix (v0.9.1)

This matrix defines the conversion from numerical Q-Scores [0, 1] to standardized technical clinical terminology used in MoeaBench certification reports.

## 1. Quality Thresholds (Tiers)

| Q-Score Range | Grade | Color Coding | Clinical Assessment |
| :--- | :--- | :--- | :--- |
| **$Q \ge 0.95$** | **EXCEPTIONAL** | Green (Solid) | Near-Ideal / Asymptotic |
| **$0.85 \le Q < 0.95$** | **HIGH** | Green (Soft) | Strong / Extensive |
| **$0.67 \le Q < 0.85$** | **STANDARD** | Yellow (Gold) | Effective / Managed |
| **$0.34 \le Q < 0.67$** | **SUBSTANDARD** | Yellow (Soft) | Partial / Limited |
| **$Q < 0.34$** | **FAILURE** | Red | Noise-dominant / Remote |

---

## 2. Dimension Granularity

| Dimension | 0.95 (Except.) | 0.85 (High) | 0.67 (Std.) | 0.34 (Substd.) | 0.0 (Fail) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DENOISE** | Near-Ideal Suppr. | Strong Suppr. | Effective | Partial | Noise-dominant |
| **CLOSENESS** | Asymptotic | High Precision | Sufficient | Coarse | Remote |
| **COVERAGE** | Exhaustive | Extensive | Standard | Limited | Collapsed |
| **GAP** | High Continuity | Stable | Managed Gaps | Interrupted | Fragmented |
| **REGULARITY** | Asymptotic Regularity | Ordered | Consistent | Irregular | Unstructured |
| **BALANCE** | Near-Ideal Balance | Equitable | Fair | Biased | Skewed |

---

## 3. Structural Markers (Marker Grammar)

- ● **Solid Circle ($Q \ge 0.5$):** Effective convergence.
- ○ **Hollow Circle ($0 \le Q < 0.5$):** Coarse convergence (Near noise floor).
- ◇ **Diamond Open ($Q < 0$):** Statistical failure (Indistinguishable from noise).

---

## 4. Analytical Summary Logic

The "Analytical Summary" is a composite sentence that identifies both high-fidelity dimensions and critical anomalies:

- **Structural Perfection**: Reserved for cases where ALL dimensions are $\ge 0.95$.
- **High Fidelity**: Lists dimensions with $Q \ge 0.85$.
- **Anomalies**: Lists dimensions with $Q < 0.67$.
- **Standard Operational Performance**: The default when no specific high/low conditions are met.
