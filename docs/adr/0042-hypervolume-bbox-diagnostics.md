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

The global non-dominated union of external reference fronts is a counterfactual diagnostic. It quantifies whether globally dominated reference points alter the reference extremes, but it never participates in the actual bounds or Hypervolume calculation.

Warnings are emitted only for structurally identifiable conditions: globally dominated reference points that actually expand at least one bound, and complete floor saturation where all final-front points of a run lie beyond the bbox. Partial outside fractions and `HV / V_bbox` are reported without interpretive thresholds.

## Consequences

- Hypervolume values retain their original numerical definition and reproducibility.
- Researchers can inspect nbox expansion, bbox occupancy, and lost contribution without hidden correction.
- A warning identifies a geometric limitation but never modifies the metric.
- Complementary metrics such as GD+ and IGD+ are preferred when Hypervolume is compressed or saturated; the bbox is not deformed to recover discrimination.
