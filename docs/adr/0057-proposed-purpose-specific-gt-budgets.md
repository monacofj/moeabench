<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0057: On-Demand, Geometry-Aware Ground Truth

**Status:** Accepted — normative decisions updated to reflect implemented source through 2026-11-23.

## Decision

MOP-specific analytical ground truth (GT) is part of the MoeaBench
public API, independently of FAIR scores. Each problem owns its
`ps(n_points)` decision-space and `pf(n_points)` objective-space
sampling rules, respecting its published mathematical definition,
constraints, component geometry, boundaries and supported dimensions.
Shared infrastructure is permitted when mathematical equivalence is
demonstrated; no universal sampler or mixed-dimensional probability
measure is imposed.

Explicit `n_points` is a **requested target**. Implementations must
validate and report actual/effective cardinality, uniqueness,
Pareto membership, component/extreme coverage, deterministic replay,
numerical precision and computational cost. A scientifically
demonstrable error in objective evaluation or GT generation is
corrected rather than preserved as normative behavior.

`mop.calibrate()` remains the explicit persistent GT operation.
`ps()` and `pf()` must not silently create disk caches or
overwrite calibrated/frozen historical references. Existing
`source_gt`, `source_search`, `source_baseline`, load/force and
sidecar contracts are preserved. Visualization may use bounded
purpose-specific sampling without silently substituting that
display sample for a calibrated scientific reference.

## Current implementation decisions

1. **DPF3:** increasing-order chaotic projection thresholds and
   numerically stable essential-objective calculations follow the
   published equations. DPF3's ordering does not change the shared
   chaotic generator used by other DPF problems.
2. **DPF4:** its *own* published projection requires ordered thresholds,
   independently of DPF3. Sort a DPF4-local copy of the threshold
   slice, without changing the shared chaotic pool or DPF1/DPF2
   behavior. Use stable complementary trigonometric calculations.
3. **DPF5:** evaluate the published P/D spherical equations, with
   strict P ownership of `x1 < 1/3`, and D ownership of the
   boundary `x1 = 1/3`. Sampling must respect each component.
4. **DTLZ8 (implemented, PR #146):** the production analytical
   `ps()`/`pf()` generator samples the true constrained Pareto
   geometry: the single line for M=2; for M>=3, a central line
   and M-1 surface patches. Its deterministic hybrid constructs
   the published components and exact requested unique objective
   count for supported budgets. At M=4 it preserves independently
   verified on-front historical points when eligible and completes
   missing points by objective-space farthest-first selection.
   Supported nonmultiple N preserves the evaluator's existing
   ignored-tail behavior. The *production source selection* is
   completed, not an unpromoted research candidate.
   **Scientific qualification remains bounded by the tested
   configurations:** pointwise Pareto membership, feasibility,
   cardinality, components and reproducibility have independent
   evidence; finite-sample coverage/regularity and relative
   line/patch weighting are configuration-specific diagnostics,
   not a universal optimality or surface-measure certificate.
5. **Public compatibility:** retain MOP evaluator and constraint
   semantics, `BaseMop.pf()` delegation, public `ps()/pf()`
   signatures and initial defaults, and explicit calibration-source
   controls. DTLZ8 currently permits N>=M with nonmultiple N,
   warning and ignoring trailing variables; GT reconstruction must
   respect that behavior unless a separately approved API change
   is adopted.

## Scientific status versus historical experiments

The production status in the numbered list above is authoritative
for implemented generators. Earlier dated research in
`docs/research/gt-issue119/` may describe experimental samplers
as provisional or earlier defects as unresolved; those statements
retain their historical meaning and **must not** be copied into
current normative status without checking merged code, source-bound
tests and the issue closure evidence.

Current architectural correctness and finite-sample suitability
are separate claims. Independent Pareto membership/feasibility
and complete component support are scientific correctness
requirements. Point density, coverage, regularity, numerical
sensitivity and cost require explicitly identified geometries,
budgets and probe/reference methods; a small empirical covering
gap does not prove optimal sampling.

## Calibration budgets and provenance

The purpose-specific, geometry-/accuracy-aware automatic
`mop.calibrate()` budget policy is accepted in principle but
**not yet specified or implemented**. Until a bounded selection
algorithm has been reviewed, the actual default remains 2,000
points for DTLZ and 10,000 for DPF (ADR 0018). Explicit user
budgets remain supported.

Reference provenance and the precise sampler policy/parameters
must be reviewable. Persisting the already-calibrated reference
inside an exported experiment remains a separate integration
decision; export may not silently generate a new GT.

## Scientific and regression requirements

For any MOP correction, maintain independent published-equation
checks, constraint feasibility, true Pareto membership, component
and boundary coverage, exact/effective cardinality, deterministic
reproduction, and relevant indicator-sensitivity diagnostics.
Mixed-dimensional coverage and regularity must be reported
component by component; a finite-probe improvement is not a
continuous-front proof or a canonical weighting policy.

**Previously existing regression tests may be updated when they
encode mathematically invalid behavior exposed by the approved
issue #119 corrections.** Such changes require explicit technical
justification and independent positive replacement checks; do
not weaken unrelated regression protection merely to obtain CI
success.

Historical frozen GT artifacts, numerical/performance baselines
and versioned certification references are separately governed by
ADR 0056. Changes to those protected artifacts require an
explicitly reviewed reference-migration decision; an updated PS/PF
implementation alone does not authorize their silent replacement.
No automatic `calibrate()` default change is authorized here.

## Relationship to absolute Hypervolume certification (ADR 0058)

This ADR governs **the MOP-owned GT's mathematical definition,
generation, calibration controls, sample budgets and provenance**.
An analytically generated GT is still finite and may underestimate
the continuous Pareto-front Hypervolume. This ADR does not promise
that `HV(GT_sample)` equals the continuous optimal volume or that
every absolute HV ratio is bounded by one.

[ADR 0058](0058-scalar-absolute-hv-certified-reference.md) is the
separate normative decision for `scale="abs"`: the original
calibrated GT remains mandatory, while the denominator is a proven
continuous-front scalar **only** for the explicitly certified built-in
problem/normalization cases, with a finite-GT fallback otherwise.
The current technical specification and applicability matrix are
in [`docs/gt.md`](../gt.md), Sections 3.4–3.5 and 9.1–9.3.
This cross-reference introduces **no new sampler, automatic budget,
sidecar schema migration or implicit calibration behavior**.

## Evidence and history

See `docs/gt.md` and `docs/research/gt-issue119/` for
mathematical derivations, negative experiments, selection tradeoffs
and migration proposals. Previous dated decisions formerly embedded
in this ADR are preserved in
`docs/research/gt-issue119/adr-0057-historical-decisions.md`.
