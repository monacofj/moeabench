# ADR 0018: Consolidated Scientific Calibration Report and Ground Truth Density Refinement

## Status

Accepted (Implemented in v0.7.6)

## Context

As MoeaBench v0.7.6 reached its final calibration phase, a peer review of the DPF4 (Degenerate Pareto Front) results revealed a significant artifact: the Hypervolume Ratio exceeded 100% (reaching ~127%). This occurred because the Ground Truth (GT) was a discrete sample of 2,000 points. Since DPF4 is a 1D curve in 3D space, the discrete sampling left "gaps" that the algorithm filled, resulting in a dominates volume greater than the discretized reference.

Furthermore, the terminology "Max Theoretical Hypervolume" was found to be misleading when applied to sampled reference sets, as it implies a continuum limit that the project does not analytically compute for all geometries.

## Technical Decision

### 1. Unified Interative Dashboard (Superseding Baseline Report)
We decided to sunset the legacy Markdown reports (`BASELINE_REPORT_v0.7.6.md`) and consolidate all scientific metrics, statistical distributions (30 runs), and visual traces into a single **Interactive HTML Calibration Dashboard**. This ensures a "Single Source of Truth" where numerical rigor and visual topology are audited together.

### 2. Metric Terminology: "Sampled Reference HV"
We abandoned the term "Max Theoretical Hypervolume" in the reports. 
*   **New Term**: **Sampled Reference HV**.
*   **Definition**: The hypervolume calculated from the discrete Ground Truth sample available in `tests/ground_truth/`.
*   **Ratio Interpretation**: Ratios > 100% are explicitly documented as **"Performance Saturation"**, occurring when an algorithm's distribution covers sampling gaps in the reference set.

### 3. Dynamic Ground Truth Density
To minimize discretization artifacts in degenerate problems:
*   **DTLZ Family (Surfaces)**: Maintained at **2,000 points**. This density is sufficient for 2D manifolds in 3D space to provide stable metrics without excessive computational cost in IGD calculations.
*   **DPF Family (Curves)**: Increased to **10,000 points**. High-density sampling is mandatory for 1D manifolds in 3D space to reduce the probability of algorithms artificially "exceeding" the reference volume by filling large gaps.

## Consequences

### Positive
*   **Numerical Accuracy**: The DPF4 "HV > 100%" artifact was reduced from ~127% to ~99.8%, providing a much more accurate representation of algorithm convergence.
*   **Institutional Memory**: The distinction between "Theoretical Limit" and "Sampled Reference" is now clearly codified, preventing future misinterpretations of metric anomalies.
*   **Efficiency**: Consolidating reports reduces repository noise and provides a superior UX for quality assurance.

### Negative
*   **Calculation Overhead**: Post-processing metrics (Phase 1B) for DPF problems now takes slightly longer (seconds vs milliseconds), though this is deemed negligible for calibration tasks.

---
*Documented in 2026-02-02 following the DPF4 Alignment Audit.*
