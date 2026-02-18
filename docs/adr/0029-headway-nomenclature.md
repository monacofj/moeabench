# ADR 0029: Headway Nomenclature and Layer 1 API Streamlining

**Status:** Accepted
**Date:** 2026-02-18
**Author:** Monaco F. J.
**Supercedes:** Sections 3.1 and 4.2 of ADR 0028.
**Drivers:** Nomenclature Clarity, API Redundancy, Scientific Precision.

---

## 1. Abstract

This Architectural Decision Record (ADR) formalizes the refinement of the *MoeaBench* Layer 1 diagnostic API. Based on user feedback and internal review of scientific presentation, we:
1.  **Rename** the physical metric `DENOISE` to **`HEADWAY`**.
2.  **Remove** the redundant `fair_` prefix from all Layer 1 function names (the "Physical Metrics").
3.  **Preserve** the `q_` prefix for Layer 2 (the "Clinical Scores") to maintain a clear cognitive distinction between raw physical facts and calibrated interpretations.

---

## 2. Decision: HEADWAY over DENOISE

While `DENOISE` accurately described the noise-mitigation aspect of the metric, **`HEADWAY`** better captures the positive directional progress of an algorithm. In the context of evolutionary multi-objective optimization (EMO), "Headway" signifies the distance gained towards the manifold, specifically in units of the problem's own resolution ($s_K$).

*   **Mapping:** `Q_DENOISE` (v0.9.1) $\to$ **`Q_HEADWAY`** (v0.9.1+).
*   **Rationale:** "Headway" is more intuitive for end-users comparing solver speed and effectiveness, while still functioning as a normalized physical measurement.

---

## 3. Decision: Removing Prefix Redundancy

The initial implementation used the `fair_` prefix (e.g., `fair_closeness`) to distinguish Layer 1 metrics from raw mathematical primitives (GD/IGD). However, since these metrics are now the primary citizen of the `MoeaBench.diagnostics` package, the prefix became redundant and verbose.

*   **Before:** `mb.diagnostics.fair_closeness()`
*   **After:** **`mb.diagnostics.closeness()`**

The distinction is now maintained via the package structure:
- `MoeaBench.metrics`: Raw mathematical primitives (Layer 0).
- `MoeaBench.diagnostics`: Physical Facts (Layer 1) and Clinical Scores (Layer 2).

---

## 4. Layer Separation Strategy

The API now follows a strict two-tier naming convention for public diagnostics:

| Layer | Type | Prefix | Example | Units |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | Physical Fact | *None* | `closeness()` | $s_K$ (Resolution Units) |
| **Layer 2** | Clinical Score | `q_` | `q_closeness()` | $[0, 1]$ (Utility) |

This allows for intuitive usage:
```python
# Measure the physical distance distribution
dist = mb.diagnostics.closeness(run_data)

# Get the clinical quality score
score = mb.diagnostics.q_closeness(dist)
```

---

## 5. Implementation and Migration

- Codebase-wide `sed` migration performed on `fair.py`, `qscore.py`, `auditor.py`, and `__init__.py`.
- Documentation and examples updated to reflect the streamlined names.
- Backward compatibility: The `fair_` prefix is no longer supported in the public `__all__` of `MoeaBench.diagnostics`.
