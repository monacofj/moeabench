<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# 17. Comprehensive Scientific Calibration and Testing Framework

Date: 2026-02-01

## Status

Accepted

## Context

MoeaBench evolved to v0.7.6 with significant improvements in MOP definitions (especially DPF family rectifications) and core architecture. However, during the transition, a critical gap emerged: **the absence of a scientifically validated baseline for performance regression testing.**

Previous tests relied on "checkouts" or visual inspections in notebooks, which are subjective and non-automatable. Furthermore, an audit revealed a severe defect in the widely used seeding strategy, where `default` arguments in constructors locked the stochastic behavior of algorithms, rendering statistical analysis void.

To mature the framework into a scientifically robust tool, we needed:
1.  **Truth**: A mathematically indisputable definition of "Optimal".
2.  **Calibration**: A systematic, reproducible baseline of what "Good Performance" looks like.
3.  **Validation**: An automated way to reject changes that degrade this performance.

## Decision

We implemented a multi-layered **Scientific Calibration and Testing Framework**, structured in three distinct phases.

### Phase 1A: The Ground Truth (The Gold Standard)

Before measuring any algorithm, we must know the target. We rejected the idea of using "very long runs" as the ground truth. Instead, we implemented **Analytical Ground Truths**:
*   For **DTLZ1-7**: Used the analytical equations to generate points directly on the manifold.
*   For **DTLZ8/9**: Due to complex constraints, we developed a **High-Fidelity Numerical Solver** (NSGA-III with Pop=1000, Gen=2000) validated against the geometric constraints ($f_M = \dots$) to produce a reference set that is guaranteed to be feasible and dense. 
*   **Result**: 2000-point CSVs frozen in `tests/ground_truth/`.

### Phase 1B: Calibration Engine (The measurement)

We built a dedicated engine (`misc/generate_baselines.py`) to map the performance of standard MOEAS (NSGA-II, NSGA-III, MOEA/D) against these truths.

**Key Architectural Decisions:**
1.  **Deterministic Hashing**: We abandoned `random.seed(time.time())`. Instead, every run uses a seed derived from its configuration: `CRC32(MOP + ALG + INTENSITY + RUN_ID)`. This guarantees that `Run05` of `DTLZ2` will *always* be the same, bit-for-bit, on any machine.
2.  **Panic Checks**: To prevent the "silent failure" of zero-variance (the fixed seed bug), the script compares the binary output of subsequent runs on the fly. If `Run01 == Run02`, it aborts immediately. 
3.  **Hybrid Intensity**:
    *   **Light** (Smoke): 100 Gen / 52 Pop. Fast (~1s). Used for CI.
    *   **Standard** (Baseline): 1000 Gen / 200 Pop. High fidelity. Used for scientific benchmarking.
4.  **Metric Isolators**: We separated generation from analysis. `compute_baselines.py` calculates metrics *post-hoc* using the saved CSVs. 
    *   **Normalization**: We normalize the Hypervolume space using the union of the **Ground Truth** and **All Observed Fronts**, creating a dynamic but consistent bounding box $[0, 1]^M$.

### Phase 2: The Testing Pyramid

We integrated these baselines into `pytest` tiers:

1.  **Fast Tier (Invariants)**:
    *   Checks mathematical properties (e.g., `sum(f) == 1` for DTLZ2).
    *   Cost: Milliseconds.
2.  **Smoke Tier (Regression)**:
    *   **Logic**: Runs a single "Light" intensity calibration.
    *   **Innovation**: Instead of a stochastic check (Mean $\pm$ 3$\sigma$), we enforce **Exact Reproducibility**. The test uses the *exact same seed* as `Run00` of the calibration. 
    *   **Assertion**: The result must match the baseline IGD/HV almost exactly. This transforms statistical regression into binary bug detection.
    *   Cost: ~1 minute for 40 tests.
3.  **Heavy Tier (Certification)**:
    *   [Planned] Runs full statistical hypothesis testing (t-test / Wilcoxon) for scientific publishing.

## Consequences

### Positive
*   **Immortality**: The project state is now frozen in `tests/baselines_v0.7.6.csv`. Any regression in the core logic will immediately break the Smoke Tier.
*   **Credibility**: We can explain exactly *why* an algorithm is performing well (metrics against analytical truth) rather than vaguely claiming "it looks good".
*   **Safety**: The "Panic Checks" and deterministic hashing prevent the recurrence of the fixed-seed methodology error that plagued previous experiments.

### Negative / Challenges
*   **Discrete Limits**: We observed negative `HV_diff` values in some perfect runs. This occurs because the 200-point MOEA population can "pack" the volume slightly better than the 2000-point discrete reference grid. This requires careful explanation to users (documented in Audit Report).
*   **Computational Cost**: Generating a full new baseline takes ~4 hours. This is acceptable for a "Major Version" release cadence but too slow for daily commits.

## References
*   `tests/baselines_v0.7.6.csv`
*   `docs/BASELINE_REPORT_v0.7.6.md`
*   `tests/test_smoke_tier.py`
