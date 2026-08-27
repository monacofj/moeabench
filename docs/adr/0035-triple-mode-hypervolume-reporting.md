# ADR 0035: Triple-Mode Hypervolume Reporting (Raw, Relative, Absolute)

## Status
Partially Superseded by [ADR 0047](0047-conventional-hypervolume-relative-scaling.md) for the operational semantics of `scale='rel'`.

## Context
Hypervolume needs an explicit normalization interpretation. Without an external reference, the evaluated experiment supplies a self-referenced bounding box. Comparisons across separate calls require the same external `ref`, which exclusively defines the bounding box.

Previously, `moeabench` implemented a dynamic normalization step where the calculated absolute volume of each algorithm was subsequently divided by the **maximum volume found in the session**. 

This forced the best algorithm to naturally hit a `1.0` ceiling. However, this caused a significant "Perspective Illusion":
1. When Algorithm A (highly stable, but perhaps suboptimal) is evaluated alone, its maximum performance is `1.0`. Due to internal variances, its average performance across runs might be reported as, e.g., `0.77`.
2. When Algorithm B (a poorly performing algorithm that explores very bad areas of space) is added to the comparison, the BBox expands massively.
3. Within this massive new BBox, the *volume* occupied by Algorithm A's average runs and Algorithm A's best runs is almost identical compared to the disastrous volume occupied by Algorithm B. 
4. The post-processing division (normalizing to `1.0`) suddenly reports Algorithm A's average as `0.91`, shifting the reported metric solely because a bad neighbor altered the scale of the box.

This mathematical behavior violates the principle of "Immutable Evaluation" for physical metrics while satisfying the desire for "Competitive Efficiency Rankings".

1. **`scale='raw'` (Volume Mode - Default Since v0.11.x)**: Returns dominated volume in the selected normalization context. A self-referenced result is not directly comparable across separate normalizations; a fixed-reference result is comparable with results using the same `ref`.
2. **`scale='rel'` (Competitive Efficiency Mode)**: Computes the absolute volume and *divides* it by the maximum volume found in the provided experiments in the current session. This forces a `1.0` ceiling based on the session winner. It answers: *"What is the competitive efficiency relative to session best?"*
    - *Note: `scale='ratio'` is preserved as a deprecated alias.*
3. **`scale='abs'` (Theoretical Optimality Mode)**: Normalizes the absolute volume by the **Ground Truth (GT)** of the underlying MOP. This requires the MOP to be pre-calibrated (via `mop.calibrate()`). It provides an absolute, cross-session score where `1.0` represents mathematical perfection. It answers: *"What is the absolute proximity to the theoretical optimum?"*

## MOP Homogeneity Validation
To prevent invalid geometric comparisons (e.g., comparing Hypervolumes across different problem topologies like DTLZ2 and ZDT1), the library now enforces MOP homogeneity:
- `raw` and `relative` scales issue a **Warning** if mixed problems are detected.
- `absolute` scale issues a **ValueError**, as normalizing against a mismatched Ground Truth produces a scientifically fraudulent score.

By making `raw` the default, `moeabench` prioritizes scientific measurement and numerical stability over competitive ranking. If a competitive ranking is desired, the user must explicitly opt-in via `scale='ratio'`.

For immutable comparisons across plotting or analysis sessions, the user must explicitly provide the same external reference front or pooled reference experiments through `ref`.

## Consequences
- **Enhanced Integrity**: The library no longer conflates physical volume with relative competitive ranking by default. 
- **Numerical Stability**: Algorithm performance metrics no longer "shift" simply because a poorly performing neighbor was added to the Bounding Box.
- **Reporting Clarity**: The compact `MetricMatrix.report()` output preserves the scale-specific framing for relative and absolute modes, while one `Reference` section identifies the concrete sources and scale-independent geometry diagnostics.

## Historical Note

The three-mode architecture (`raw`, `rel`, `abs`) established here remains the basis of conventional Hypervolume reporting. The historical `rel` text above, including the session-maximum `1.0` ceiling and the `ratio` alias, is preserved as a record of the v0.11 decision but is not the current contract. [ADR 0047](0047-conventional-hypervolume-relative-scaling.md) now defines `H_rel`: one fixed best-final denominator in the selected reference context, individual reference fronts rather than a pooled union, and no clipping. The canonical public scale name is `rel`.
