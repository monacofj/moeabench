<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0047: Conventional Hypervolume Relative Scaling

## Status

Accepted

## Date

2026-08-27

## Context

ADR 0019 introduced the tripartite Hypervolume vocabulary and ADR 0035 later
formalized `raw`, `rel`, and `abs` reporting. Their historical description of
relative Hypervolume treated the denominator as a session maximum and therefore
implied a `1.0` ceiling. That description is no longer sufficient once callers
supply an explicit external reference context.

Reference geometry and relative scaling are separate decisions. The selected
reference determines the common ideal/nadir bounds and Hypervolume reference
point; `scale="rel"` is applied only after raw Hypervolume has been evaluated in
that fixed geometry.

## Decision

For conventional Hypervolume, `scale="rel"` uses one fixed denominator for the
entire returned trajectory.

- Without explicit `ref`, the evaluated runs define one self-reference context.
  Each run's final front is evaluated in that common geometry and the largest
  final Hypervolume is the denominator.
- With explicit `ref`, ideal/nadir bounds are derived exclusively from the
  supplied reference context. Each individual final reference front is then
  evaluated separately in those same bounds, and the largest of those final
  Hypervolumes is the denominator.
- The denominator is never the Hypervolume of the pooled union of the reference
  fronts. Pooling could construct a super-front that no individual reference
  attained.
- The same scalar denominator is used for every generation and run in the
  returned matrix. Generation-wise rescaling is not permitted.
- Relative values are not clipped. `H_rel = 1.0` means equality with the chosen
  denominator; values above `1.0` are valid when evaluated data outperform an
  external reference, or when an earlier trajectory state exceeds the selected
  final-reference denominator.
- Relative scaling does not modify the nbox, bbox, raw Hypervolume, engine
  selection, or reference diagnostics. Those remain governed by ADR 0020,
  ADR 0042, and ADR 0043.
- `H_rel` is a contextual competitive scale, not a statement of theoretical
  optimality. Ground-Truth normalization remains the role of `scale="abs"`.
- The canonical public conventional-HV scales are `"raw"`, `"rel"`, and
  `"abs"`. Historical `"ratio"` terminology is not part of the current
  canonical contract.

Exact and Monte Carlo denominator evaluation follows the same Hypervolume
backend policy and reproducibility settings used by the evaluated trajectory.

## Consequences

- Separate calls are directly comparable only when they use the same explicit
  reference context.
- An external reference defines a ruler, not a ceiling.
- A relative value above one is informative and must not be silently clipped.
- Raw and relative Hypervolume retain identical geometric diagnostics; only the
  reported scale differs.
- ADR 0019 and ADR 0035 remain historical records of the tripartite design but
  are partially superseded by this ADR for the operational semantics of
  `H_rel`.

## Relationship to Earlier Decisions

- [ADR 0019](0019-tripartite-hypervolume-metrics-nomenclature.md) introduced the
  three Hypervolume perspectives.
- [ADR 0020](0020-standardized-reference-point-strategy-for-normalization.md)
  defines normalized Hypervolume reference geometry.
- [ADR 0035](0035-triple-mode-hypervolume-reporting.md) established the
  `raw`/`rel`/`abs` reporting surface.
- [ADR 0042](0042-hypervolume-bbox-diagnostics.md) makes bbox diagnostics
  descriptive rather than corrective.
- [ADR 0043](0043-hypervolume-engine-selection.md) defines exact/Monte Carlo
  engine selection and reproducibility.
- [ADR 0046](0046-relative-ordinal-hypervolume.md) specifies analogous relative
  scaling for OHV, but OHV retains its separate ordinal geometry.
