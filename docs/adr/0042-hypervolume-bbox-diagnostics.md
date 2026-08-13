<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0042: Diagnostic-Only Policy for Hypervolume Bounding Boxes

## Status

Accepted

## Context

A fixed Hypervolume reference can lose discrimination when dominated reference points stretch its geometry or evaluated fronts lie beyond its usable region. Correcting that geometry automatically would change the scientific question and make results depend on undocumented data treatment.

The normalization box (**nbox**) spans `ideal` to `nadir`. The Hypervolume bounding box (**bbox**) spans `ideal` to the normalized reference point 1.1. The two terms are not interchangeable.

## Decision

Hypervolume geometry remains fixed by the selected context. Without `ref`, bounds come from the evaluated experiment. With `ref`, bounds come exclusively from the complete external reference, including any outliers or dominated points the user supplied.

MoeaBench performs diagnostic observation only:

- no automatic bbox correction or construction;
- no outlier removal, clipping, trimming, percentile, IQR, or z-score policy;
- no problem- or algorithm-calibrated replacement reference;
- no use of evaluated points to expand external bounds;
- no threshold for declaring Hypervolume values too similar.

Each supplied reference item is treated as one diagnostic source. MoeaBench first computes the non-dominated union within each source, then compares the objective-wise extent of their union with the globally non-dominated envelope. The resulting **global/local ND coverage** is

\[
C_j = \frac{\operatorname{range}(G_j)}{\operatorname{range}(L_j)}.
\]

Here, \(L\) is the union of the locally non-dominated sources and \(G=ND(L)\). A value of 1 means that the global envelope retains the full local extent in objective \(j\); a value below 1 means that part of that extent exists only locally. When both spans are zero, coverage is defined as 1. This is descriptive and does not classify the Hypervolume as valid or invalid.

The **reference expansion** relative to the fronts represented by the final reported row is

\[
E_j = \frac{\operatorname{range}(F_{reference,j})}{\operatorname{range}(F_{reported,j})}.
\]

A value of 1 means no expansion, a value above 1 means that the reference is wider in that objective, and infinity means that the reported result has zero span while the reference does not. Expansion does not automatically imply compression or another defect.

These diagnostics never participate in the actual bounds or Hypervolume calculation. Complete bound vectors remain available through `MetricMatrix.diagnostics`, but the standard report presents only affected objectives so that it remains usable for many-objective problems. Existing geometry keys remain available for compatibility.

Coverage, expansion, and dominated-reference fractions do not emit warnings. A structural warning is emitted only for complete floor saturation where all final-front points of a valid run lie beyond the bbox. Partial outside fractions and `HV / V_bbox` are reported without interpretive thresholds.

## Consequences

- Hypervolume values retain their original numerical definition and reproducibility.
- Researchers can inspect local/global coverage, reference expansion, bbox occupancy, and lost contribution without hidden correction.
- The floor-saturation warning identifies a structural limitation but never modifies the metric.
- Complementary metrics such as GD+ and IGD+ are preferred when Hypervolume is compressed or saturated; the bbox is not deformed to recover discrimination.
