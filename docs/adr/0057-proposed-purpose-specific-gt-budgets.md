<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0057: Purpose-Specific Budgets for On-Demand Ground Truth

## Status

**Accepted in principle — 2026-11-19; targeted DPF3/DPF5 equation and
sampler corrections authorized by 2026-11-20 addendum below; automatic
calibration budget and historical-reference migration still pending.**
The maintainer approved on-demand MOP-specific GT generation, the separation
between `pf()` and `calibrate()`, and automatic budget selection for
`calibrate()` when `n_points` is omitted. The original decision did not
authorize implementation; the dated addendum below now authorizes **only**
the listed DPF3/DPF5 equation and sampler work. The general numerical budget
policy, universal quality tolerance, automatic calibration defaults,
historical baseline replacements and preexisting regression-test edits remain
**unapproved**. Tracked in [#119](https://github.com/monacofj/moeabench/issues/119)
under umbrella [#122](https://github.com/monacofj/moeabench/issues/122).

**Scope of ADR 0018 supersession:** its fixed 2,000-DTLZ / 10,000-DPF
budgets are no longer the intended universal policy. Until the new
numerical default algorithm is reviewed and implemented, these remain
the *actual runtime defaults* and accepted historical calibration
reference values. The other decisions in ADR 0018 are untouched.
No change to existing data, tests or certification follows merely
from accepting this direction.

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

## Accepted architectural direction (implementation pending)

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
   generation at a user-chosen budget. Scientific calibration with no explicit `n_points` **selects an
   appropriate budget automatically** using MOP geometry, desired
   accuracy and bounded computational cost. The exact selection rule,
   budget values and maximum work budget remain to be defined and
   validated before implementation.
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

## Details requiring review before implementation

- Which geometry-/quality-aware automatic budget rule will be used by
  `mop.calibrate()` when `n_points` is omitted? What are its cost bound,
  convergence/coverage tolerance, documented limitations and migration
  behavior? Determining the rule, not choosing between required or
  automatic budget, is the remaining research question.
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

## Consequences and transition

The adopted *direction* replaces ADR 0018's assumption that a single
fixed DTLZ/DPF density is appropriate for every scientific use. The
existing 2,000/10,000 numerical defaults and frozen references remain
unaltered **until a separately reviewed, concrete policy is implemented**.
ADR 0018's other historical observations and decisions remain intact.
Validation results and explicit sign-off precede any code, regression,
baseline, certification or API changes.

## Implementation

**Originally none to the runtime.** This section records the initial
2026-11-19 accepted direction. The **2026-11-20 authorization addendum
below** now permits narrowly scoped DPF3/DPF5 corrections in Draft PR
#140, without authorizing a new `calibrate()` default, persisted GT
replacement, existing regression modification or certification rewrite.


## Implementation Authorization Addendum — 2026-11-20 (issue #119, PR #140)

The maintainer has now **explicitly authorized work on the DPF3/DPF5
corrections and updating ADRs to reflect the decisions made in this
conversation**. This dated addendum records that decision without
retroactively rewriting the original status or the unapproved numerical
calibration policy.

**Authorized implementation work in Draft PR #140:**

- Correct the DPF3 evaluator to the published **increasingly sorted**
  projection thresholds, using numerically stable essential-objective
  calculations. Sorting is **DPF3-specific**; it must not silently change
  the chaotic sequence used by DPF1/DPF2/DPF4.
- Correct the DPF5 published P and D spherical evaluation equations for
  supported dimensions, including the strict P condition `x1 < 1/3`
  and D ownership of `x1 = 1/3`.
- Implement **separate, geometry-aware PS/PF finite sampling for each
  MOP**. Shared kernels, random sequence engines, storage and validation
  infrastructure are permitted only where their scientific applicability
  is demonstrated. Neither a universal MOP sampler nor a fixed
  mixed-dimensional P/D probability law is adopted.
- Preserve the explicit `n_points` argument as a **requested sample
  budget**, transparently reporting and testing returned/effective
  cardinalities. Verify Pareto membership, component/extreme coverage,
  deterministic behavior and numerical quality independently.
- Add new independent tests and auditable numerical comparisons. Keep
  objective-evaluator changes and finite GT/reference migration
  separately attributable.

**Not authorized by this addendum:**

- A replacement for the historical automatic `calibrate()` default
  budgets (the existing 2,000/10,000 runtime rules remain in effect),
  a particular target epsilon, a universal P/D allocation law or
  a new unattended persistence/cache contract.
- Replacing any frozen GT, performance reference, certificate or
  baseline merely to accommodate changed outputs. ADR 0056's
  versioned-reference lifecycle remains binding.
- Editing an **existing** regression test or fixture without advance
  warning and a further explicit maintainer approval. An old reference
  mismatch is a migration decision, not a test failure to suppress.

**Scientific evidence and staged implementation:** the proofs and
negative experiments are documented in
[docs/gt.md](../gt.md) and
[research/gt-issue119/](../research/gt-issue119/README.md).
Draft PR #140 may implement and validate the above corrections, but it
must not be merged with failing historical stability gates or promote
a sampler to a fully certified GT merely on the basis of finite-probe
statistics. A separately accepted, versioned migration resolves any
incompatibility with the historical DPF5 numerical baseline.

The original "implementation pending" wording above records the
decision's status *before this addendum*. The explicit authorization
here is limited to the described DPF3/DPF5 corrections; automatic
calibration budgeting remains pending.
