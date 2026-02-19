# ADR 0031: Numerical Integrity Verification Tier (v0.9+)

## Status
Proposed/Accepted

## Context
As MoeaBench reaches maturity with the v0.9+ diagnostic framework, ensuring numerical reproducibility across different environments and code iterations has become a primary requirement. While stochastic convergence tests (Smoke Tier) are necessary, they are not sufficient to detect subtle regressions in the clinical diagnostic pipeline (e.g., Q-Score calculations, baseline resolution, or performance optimization flags).

Furthermore, the introduction of a caching mechanism in `baselines.py` to resolve performance bottlenecks (re-parsing large JSON files) introduced a new stateful component that required strict verification.

## Decision
We implement a dedicated **Numerical Integrity Verification Tier** (Regression Tier) focused on bit-exact (or high-precision) reproducibility of clinical diagnostics.

### 1. The Stratum
- The tier uses a fixed, high-precision reference dataset (`calibration_reference_audit_v0.9.json`) and a set of pre-calculated target values (`calibration_reference_targets.json`).
- It verifies that the current `auditor.audit()` output matches these targets down to 6 decimal places.
- This tier is positioned between the functional **Unit Tests** and the stochastic **Smoke Tests**.

### 2. Performance Optimization & Caching
- To enable this tier to run efficiently, we implemented an in-memory caching mechanism in `MoeaBench/diagnostics/baselines.py`.
- A `_CACHE_DIRTY` flag ensures that baseline sources are only merged and parsed once, unless new sources are registered during runtime.
- This optimization reduces the audit time for 42 scenarios from ~20s (with redundant I/O) to ~2s.

### 3. Default Execution (The Testing Pyramid)
- Given its speed (~2s) and high analytical value, the **Regression Tier** is promoted to the **Default Test Run**.
- Executing `python3 test.py` now automatically includes:
    1. **Unit Tests** (Foundation)
    2. **Light Tier** (Geometric Invariants)
    3. **Regression Tier** (Numerical Integrity)

## Consequences
- Developers receive immediate feedback on whether their changes "broke the math" of the diagnostics.
- The system prevents performance regressions by ensuring baseline loading remains optimized.
- Numerical drift in sensitive clinical metrics (EMD, JSD, ECDF) is caught early in the development lifecycle.
- Nomenclature is standardized away from "Golden Values" toward "Calibration Reference Data" to maintain professional rigor.
