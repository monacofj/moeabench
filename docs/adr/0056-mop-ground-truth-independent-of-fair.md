<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0056: MOP Ground Truth Independent of FAIR for Absolute Hypervolume

## Status
Accepted

## Date
2026-11-13

## Context
FAIR research must be separable from the MOP reference capabilities of MoeaBench.
The 0.17.0 cleanup incorrectly required an explicit `ref` for absolute HV
and used that `ref` as the denominator, conflating normalizing geometry
with the mathematical Pareto front of the underlying problem.

Built-in MOPs already supply `ps(n_points)` and `pf(n_points)`, exposed
also as `exp.optimal()` and `exp.optimal_front()`. These are separate from
FAIR's Q-score baselines and calibration registry.

## Decision
- Restore `mb.metrics.hv(exp, scale='abs')` for experiments, runs and
  sourced populations with an associated MOP.
- Sample GT through `mop.pf(n_points=500)`, using the documented
  analytical sampling default pending separate convergence/density studies.
- Compute `H_abs(P) = HV(P) / HV(GT_MOP)`, with identical normalization
  bounds, backend and Monte Carlo seed in numerator and denominator.
- `ref` remains optional and **exclusively** determines external
  normalization bounds. Multiple reference sources are allowed. It does
  not replace or otherwise select GT; when omitted, preserve the
  shared self-reference bounds.
- Reject missing MOP provenance, unimplemented `pf()`, empty, nonfinite,
  dimensionally incompatible, or nonpositive-HV GT. Do not silently
  fall back to raw HV.
- Preserve `raw` and `rel` semantics without alteration.

## Scientific Qualifications
A sampled finite GT approximates a continuous front. Results above one
can arise from sampling density or geometry and are not evidence of
surpassing the true Pareto optimum. We expose the GT source, sampled point
count and GT-HV denominator as diagnostics. The default 500-point policy
is provisional and should be tested for sensitivity, particularly on
degenerate and many-objective problems, before 1.0.0.

## Consequences
Restore the established absolute-HV user API without FAIR calibration.
Maintain the historical 0.17.0 changelog as published and document the
correction under Unreleased. Restore the independent Example 15.

## Related ADRs
- [0015](0015-scientific-ground-truth-and-numerical-integrity.md): MOP GT sampling.
- [0019](0019-tripartite-hypervolume-metrics-nomenclature.md),
  [0035](0035-triple-mode-hypervolume-reporting.md): triple HV perspectives.
- [0047](0047-conventional-hypervolume-relative-scaling.md): contextual rel mode.
