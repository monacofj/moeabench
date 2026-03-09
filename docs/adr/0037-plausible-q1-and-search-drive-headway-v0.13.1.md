# ADR 0037: Plausible Q1 and Search-Drive Headway (v0.13.1)

**Status:** Accepted
**Date:** 2026-03-08
**Author:** Monaco F. J.
**Supercedes:** Physical layer definitions in ADR 0029 and ADR 0036.
**Drivers:** Clinical Fairness, Semantic Consistency, Finite-Resolution Integrity.

---

## 1. Abstract

This ADR formalizes two critical refinement steps in the *moeabench* diagnostic architecture for version `v0.13.1`:
1.  **Plausible Q1 Correction**: The physical `CLOSENESS` metric now subtracts the median discretization residue ($ideal\_res$) to ensure perfect algorithms can achieve $Q=1.0$ even with finite Ground Truths.
2.  **Search-Drive Headway**: The physical `HEADWAY` metric is redefined as an adimensional ratio of the residual search error relative to a random search baseline, eliminating its previous redundancy with `CLOSENESS`.

---

## 2. Decision: Plausible Q1 (Closeness Residue)

### 2.1 The "Finite GT" Penalty
Previously, even a mathematically perfect optimizer could not achieve a physical `CLOSENESS` of 0.0 because the Ground Truth (GT) is a discrete point set. The "holes" in the GT created a persistent distance floor (discretization residue). In v0.13.1, we introduce an empirical correction at the physical layer.

### 2.2 Mechanism
- During baseline calibration, we sample a fresh analytical ideal front and measure its median distance to the discrete GT. This value is stored as `ideal_res`.
- The physical `CLOSENESS` metric is now calculated as:
  $$ u_{corrected} = \max(0, u_{raw} - ideal\_res) $$
- **Result:** Algorithms are no longer penalized for the finite resolution of the benchmark itself.

---

## 3. Decision: Search-Drive Headway (Adimensional)

### 3.1 Resolving Redundancy
In previous versions, `HEADWAY` was simply a 95th-percentile version of `CLOSENESS`, both measured in resolution units ($s_K$). This created a redundant diagnostic overlap in the physical layer.

### 3.2 Redefinition as "Residual Search Error"
We redefine the physical `HEADWAY` as the fraction of search error that remains relative to the null hypothesis (chaos):
$$ \mathrm{HEADWAY} = \frac{GD_{95}(P \to GT)}{GD_{95}(\text{Random} \to GT)} $$

- **Ideal (0.0):** Search successfully reached the manifold.
- **Random (1.0):** No progress beyond a random guess.
- **Meaning:** It measures the "Search Drive"—how much of the initial entropy was dominated by the optimizer.

---

## 4. Layer Units and Rules

Following v0.13.1, the physical layer metrics follow different unit paradigms:

| Metric | Unit Paradigm | Rationale |
| :--- | :--- | :--- |
| `CLOSENESS` | $s_K$ (Resolution) | Measures microscopic precision. |
| `HEADWAY` | Adimensional Ratio | Measures global search progress. |
| `COVERAGE` | Objective Space | Measures macroscopic extent. |
| `BALANCE` | Divergence (JS) | Measures distributional entropy. |

---

## 5. Migration and Versioning

- The diagnostic baseline is updated to **`baselines_v0.13.1.json`**.
- All regression targets are updated to reflect the new physical and clinical scales.
- Visual report scripts (`audit_calibration.py`, `generate_visual_report.py`) are updated to support the new metadata (notably `ideal_res`).
