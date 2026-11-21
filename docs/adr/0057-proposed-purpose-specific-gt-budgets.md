<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0057: On-Demand, Geometry-Aware Ground Truth

**Status:** Accepted — current policy as of 2026-11-21.

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
4. **DTLZ8:** replace mathematically invalid GT sampling with
   deterministic analytical sampling of its true Pareto geometry:
   the single-constraint line for M=2 and, for M>=3, a central
   line plus M-1 surface patches. A geometry-aware hybrid is the
   current research candidate for M=3/5/10; M=4 may preserve
   independently verified on-front incumbent points and complete
   missing points by objective-space farthest-first selection.
   **Candidate selection requires component-wise validation before
   being promoted as certified GT**, including arbitrary supported
   M and nonmultiple N behavior.
5. **Public compatibility:** retain MOP evaluator and constraint
   semantics, `BaseMop.pf()` delegation, public `ps()/pf()`
   signatures and initial defaults, and explicit calibration-source
   controls. DTLZ8 currently permits N>=M with nonmultiple N,
   warning and ignoring trailing variables; GT reconstruction must
   respect that behavior unless a separately approved API change
   is adopted.

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

## Evidence and history

See `docs/gt.md` and `docs/research/gt-issue119/` for
mathematical derivations, negative experiments, selection tradeoffs
and migration proposals. Previous dated decisions formerly embedded
in this ADR are preserved in
`docs/research/gt-issue119/adr-0057-historical-decisions.md`.
