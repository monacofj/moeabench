# Audit Report: MoeaBench Diagnostics Compliance

**Date:** 2026-02-10  
**Target Specification:** `MoeaBench_Especificacao_Unificada_Diagnostics.md`  
**Verdict:** 🔴 **NON-COMPLIANT**

## 1. Executive Summary

The audit reveals a standard-compliant implementation for the **FIT (Proximity)** dimension, which strictly follows the "Metrological Sanitization" and "ECDF Q-Score" requirements.

However, the codebase fails to adhere to the specification's implicit mandate for **Global ECDF Standardization**. While the specification header "5. Q-score melhorado por ECDF (obrigatório)" strictly mandates the ECDF approach, the current implementation falls back to legacy **Linear Interpolation** for all other dimensions (Coverage, Gap, Regularity, Balance). Consequently, the baseline generation scripts are also incomplete.

## 2. Compliance Matrix

| Component | File | Status | Critical Defects |
| :--- | :--- | :--- | :--- |
| **Q-Score Logic** | `MoeaBench/diagnostics/qscore.py` | 🔴 **FAIL** | Uses Linear Interpolation for COV, GAP, REG, BAL. Violates Section 5 (ECDF mandatory). |
| **Baselines Generator** | `tests/calibration/generate_baselines_v4.py` | 🔴 **FAIL** | Only generates ECDF data for FIT. Missing ECDF distributions for other metrics. |
| **Baseline Data** | `MoeaBench/diagnostics/resources/baselines_v4.json` | 🔴 **FAIL** | JSON schema is valid but incomplete (missing non-FIT ECDF data). |
| **Auditor Engine** | `MoeaBench/diagnostics/auditor.py` | 🟢 **PASS** | Correctly orchestrates the pipeline. |
| **Visual Report** | `tests/calibration/generate_visual_report.py` | 🟡 **PARTIAL** | Correctly implements the "Biopsy" (CDF) view, but the Q-Scores displayed in the "Verdict Matrix" are derived from the non-compliant linear method for non-FIT metrics. |

## 3. Detailed Findings

### 3.1. Violation of Q-Score Standardization (Section 5)
**Requirement:** Section 5 states "Q-score mejorado por ECDF (obrigatório)". It describes a rigorous probabilistic scoring method ($P(X < x)$) based on an empirical distribution ($N=200$).
**Finding:**
- `qscore.compute_q_fit`: **Compliant**. Uses `_compute_q_ecdf`.
- `qscore.compute_q_coverage`: **Non-Compliant**. Uses `_compute_q_linear`.
- `qscore.compute_q_gap`: **Non-Compliant**. Uses `_compute_q_linear`.
- `qscore.compute_q_regularity`: **Non-Compliant**. Uses `_compute_q_linear`.
- `qscore.compute_q_balance`: **Non-Compliant**. Uses `_compute_q_linear`.

**Impact:** The "Unified Specification" aims to remove arbitrary scales. Linear interpolation preserves arbitrary scaling issues for 4 out of 5 metrics, undermining the certification's scientific validity.

### 3.2. Incomplete Baseline Generation (Section 3 & 6)
**Requirement:** Determining the `rand_ecdf` and `uni50` (or equivalent) for all metrics to support the Q-Score.
**Finding:**
- `generate_baselines_v4.py` calculates `rand_ecdf` **only for the FIT metric**.
- For COV, GAP, REG, and BAL, it relies on legacy scalar values (Mean/Median) suitable only for linear interpolation, not the required ECDF distribution.

### 3.3. Specification Gap
**Observation:** The Unified Specification file (`MoeaBench_Especificacao_Unificada_Diagnostics.md`) explicitly details the *sanitization of FIT* but is silent on the specific "Physics" (Unit Definitions) for COV, GAP, REG, and BAL.
**Risk:** While the *scoring method* (ECDF) is mandated, the *input metric definitions* for these dimensions are not in the spec file. The code currently uses "Fair Coverage", "Fair Gap", etc., which exist in `fair.py` (not audited here but referenced).

## 4. Required Corrections

To achieve compliance, the following actions are mandatory:

1.  **Refactor `generate_baselines_v4.py`**:
    - Must generate `rand_ecdf` (list of 200 sorted values) for **ALL** 5 dimensions (FIT, COV, GAP, REG, BAL).
    - Must computed `uni50` (Ideal reference) for all dimensions where applicable.

2.  **Refactor `qscore.py`**:
    - Remove `compute_q_linear` usage for clinical metrics.
    - Wire `compute_q_coverage`, `compute_q_gap`, `compute_q_regularity`, and `compute_q_balance` to use `_compute_q_ecdf`.

3.  **Regenerate Baselines**:
    - Run the updated generator to produce a complete `baselines_v4.json`.
