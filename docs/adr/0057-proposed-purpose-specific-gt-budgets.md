<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0057 (Proposed): Purpose-Specific Budgets for On-Demand Ground Truth

## Status

**Proposed — NOT ACCEPTED.** Discussion in issue
[#119](https://github.com/monacofj/moeabench/issues/119) under umbrella
[#122](https://github.com/monacofj/moeabench/issues/122).
Do not implement this proposal or treat it as a supersession before
explicit maintainer approval of the final contract.

## Existing accepted decisions and behavior

- ADR 0006 provides analytical `ps()`/`pf()` at the MOP level.
- ADR 0016 evolves analytical and guided geometric PF construction.
- ADR 0017 treats GT as a scientific reference/artifact, including
  calibrated/frozen references and independent testing. This proposal
  **does not reintroduce GT as a new concept**.
- ADR 0018 set **2,000 DTLZ / 10,000 DPF** calibration densities to
  address the then-measured sampling artifacts. It also records
  historical calibration/reporting decisions beyond point budgets.
- As implemented, `BaseMop.pf(n_points=100)` delegates to `ps()`;
  `exp.optimal_front(n_points=500)` generates via `ps()` and evaluation;
  `view.topology(exp)` implicitly uses that 500-point path;
  `mop.calibrate()` defaults to ADR 0018's 2,000/10,000 and writes
  an explicit GT sidecar with `gt_reference`; existing GTs may also be
  loaded from external sources or obtained empirically.
- An exported experiment has existing selective `all`, `config` and
  `data` modes (ADR 0011). Explicit preservation of the exact GT used
  in an experiment remains an integration question; do not assume it
  already has a separate contract or change persistence here.

## Problem

Fixed point budgets based only on DTLZ/DPF family do not control
coverage error uniformly across objective dimensions, intrinsic
dimensions and disconnected/mixed-dimensional geometries. Users can
select varied M, N, K and D; precomputed references for all such
combinations are infeasible. Visualization has a different point
budget and latency objective than numerical metrics/calibration.

The adopted ADR 0018 budgets served a *historical calibration use
case*, but are not a universal theorem of GT sufficiency. Altering
that policy requires a **reviewed partial supersession** of ADR 0018;
its other approved historical decisions must not be silently erased.

## Proposed direction (decision pending)

1. **On-demand by MOP/configuration:** preserve MOP-owned analytical
   evaluation and generation; prefer deterministic, scientifically
   validated geometric algorithms suited to the exact MOP, objective
   and decision dimensions and constraint configuration. Some
   geometries admit fast arc-length/structured rules; others may
   need numerical optimization. Do not precompute a favored list of
   M values and imply other dimensions are less valid.
2. **Explicit budgets:** users may pass a requested point budget.
   Interpret it as a *requested target*, auditing whether a generator
   returns the requested number and whether filtering reduces it.
   Where an actual count differs, expose/document it; do not silently
   claim exact cardinality. No universal fixed DTLZ/DPF count.
3. **Purpose-specific defaults:** keep lightweight, bounded
   visualization sampling; allow more rigorous calibration/reference
   generation at a user-chosen budget. Whether scientific calibration
   with no supplied budget (a) requires explicit `n_points` or
   (b) computes a geometry-dependent default is **undecided**.
   Also undecided: numerical defaults and maximum work budget.
4. **Scientific quality:** certify complete Pareto membership,
   front-region coverage, point spacing/regularity and convergence
   independently of any FAIR metrics. Keep geometry correctness
   separate from discretization quality and numerical resolution.
   Independent dense samples are diagnostics, not automatically the
   canonical definition of an analytically known front.
5. **Persistence:** existing `mop.calibrate()` is the explicit
   persistent GT path. Do **not** introduce implicit on-disk caching
   for every `ps()`/`pf()` invocation. A separate future task
   should investigate embedding the *actual calibrated GT used* in
   exported experiments, without silently generating it during
   export, observing selective save/load contracts.
6. **Plotting:** sample/reduce for the display only, preserving
   extremes and relevant components; do not force all calibrated GT
   points into interactive plots or substitute a display decimation
   as the metric's scientific reference.

## Questions requiring approval before implementation

- Exact behavior of `mop.calibrate()` without `n_points`:
  required explicit budget, geometry-dependent default, or another
  defined policy? What is the migration behavior?
- Are `ps()` and `pf()` public defaults to remain backward-compatible
  initially? How will cardinality and supported configurations be
  reported?
- Where/how are reference provenance and point-generation parameters
  recorded? Should GT artifacts be carried in `exp.save(mode='all')`,
  and what does `mode='data'` mean?
- Which changes require an explicit public/API or version transition,
  and which legacy fixtures/baselines (if any) would need
  maintainer-authorized replacement?
- Which geometric error bounds, cost limits, and reproducibility
  requirements are sufficient to promote individual samplers?

## Consequences if accepted

The default DTLZ/DPF calibration densities in ADR 0018 would be
**partially superseded only as specified by an approved concrete
budget policy**. Its original observations and other decisions remain
historical/normative as applicable. Validation results and explicit
sign-off precede any code, regression, baseline, certification or
API changes.

## Implementation

**None.** This is a proposal for review, not a merged architecture
change. No changes to sampling defaults, calibrated GT storage,
visualization, existing regression tests or certification references
are authorized here.
