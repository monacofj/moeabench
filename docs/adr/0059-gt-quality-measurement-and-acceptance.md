<!--
SPDX-FileCopyrightText: 2026 Monaco F. J.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0059: Measurable GT Quality and Independent Acceptance

## Status

**Accepted (maintainer authorization 2026-11-23).** The initial GT
assessment API, MOP report integration, and independent objective-space
oracles listed below are implemented. This accepted decision does
**not** claim that all GTs have obtained a release certificate.

## Decision and scientific scope

Ground Truth (GT) validity and finite-sampling suitability are
**distinct scientific properties**. MoeaBench shall make observable
quantities accessible through a structured public API and protect
supported invariants through independent tests.

- **Mathematical correctness:** true Pareto-front membership under
  independently derived problem equations, decision-space feasibility
  when a decision witness exists, and component/boundary support.
- **Finite-sample quality:** effective cardinality, uniqueness,
  geometrical coverage/resolution, within-component spatial
  regularity, reproducibility, indicator sensitivity and bounded
  measurement cost.
- **Provenance and limits:** the original MOP, reference origin,
  objective/decision dimensions, numerical tolerance, probe budget,
  chosen geometry and independent oracle.

An empirical covering gap is **not** a continuous Hausdorff certificate.
A finite Pareto-valid GT can still give biased IGD/IGD+/HV results.
Quality observations without a scientifically justified tolerance
must not be silently classified as a passing quality certificate.
No single universal weighted GT quality scalar is approved.

## Implemented API and reporting

```python
audit = mop.assess_gt(
    reference=None,
    n_points=None,
    probes=128,
    seed=0,
    tolerance=1e-9,
)
print(audit.summary())
print(audit.to_dict())
print(audit.validity, audit.coverage, audit.regularity)
mop.report()                    # includes audit of calibrated in-memory GT
mop.report(assess=False)        # brief report without diagnostics
```

`reference` accepts an explicit objective-space matrix; an explicitly
supplied `n_points` requests a **read-only** analytical `pf()`
candidate. They cannot be supplied simultaneously. When both are
omitted, the method inspects the already calibrated **in-memory** GT,
which must have been created or loaded by an explicit `calibrate()`.
Neither `assess_gt()` nor `report()` implicitly loads or writes
sidecars, calibrates a MOP, changes the GT or global RNG, or modifies
any frozen certification reference.

`GTAssessment` is a frozen structured result containing immutable
`GTProperty` values, `summary()` and JSON-serializable
`to_dict()`. Every property exposes `status`, `value`,
`basis`, `threshold` and `unit`. Allowed statuses are
`pass`, `fail`, `unverified`, `not_applicable`.

The report follows MoeaBench's ordinary `Reportable` conventions
(`show`, Markdown/plain presentation); when no GT is calibrated
it reports the absence clearly **without** making a hidden GT.
The report displays numerical measurements and unsupported claims
with their actual status rather than asserting scientific approval.

## Implemented independent objective-space validators

These validators operate only for **unmodified original built-in**
MOP classes. A subclass, instance with an overridden `ps`,
`pf` or `evaluation`, or custom MOP does not inherit a false
published-geometry certificate.

| Original MOP | Independent checked relation |
|---|---|
| DTLZ1 | Nonnegative essential simplex, sum of objectives 0.5; axis extrema |
| DTLZ2–4 | Nonnegative unit hypersphere; axis extrema |
| DTLZ7 | Exact disconnected Pareto-record intervals and final objective; record-component occupancy |
| DTLZ8 | Published constrained single line or central-line-plus-patches; per-patch occupancy |
| DTLZ9 | Published constrained objective arc and exact endpoints |
| DPF1 | Essential simplex and independently recomputed redundant linear projections |
| DPF2 | Essential Pareto-record domain and independent squared redundant projections; record-component occupancy |
| DPF3–4 | Inverted essential sphere and independently reconstructed ordered nonlinear projection coordinates |
| DPF5 | Published positive-unit-sphere P/D branches and D-branch repeated coordinates; both-branch occupancy |

DTLZ5/6 (including genuine `g>0` Pareto regions at `M>=4`)
**do not** inherit a DTLZ2 spherical certificate. The independent
objective-space geometry audit now verifies the original angular
feasible-image constraints, mathematically necessary minimum radial
`g`, and the global Pareto membership of exact `g=0` vectors
(by the global minimum feasible Euclidean norm). A point violating
the feasible-image or minimum-radius conditions is `fail`.
A feasible positive-`g` point without an independent global
dominance exclusion is **`unverified`**, even when its necessary
geometry checks pass. For `M>=4`, the **whole-GT validity status stays `unverified`** even
when a supplied reference contains only individually proved `g=0`
points: those points alone do not cover the original complete PF.
A mixture of proved `g=0` points and such positive-`g` points
is likewise `unverified` as a whole.
Separate counts show both parts of the corrected original reference.
At hyperspherical coordinate singularities, `atan2(0,0)`
does not define a unique angular coordinate. The independent
feasible-image validator must choose a realizable angle inside
the original admissible interval (the midpoint `pi/4`)
rather than rejecting the original last-axis Pareto
endpoint `(0,...,0,1)`. This implements the proven
minimum-angular-distortion construction documented
in [`docs/gt.md`](../gt.md), Section 3.6.2.
Positive-`g` points on the identical singular ray
remain radially dominated; truly determined forbidden
angles must continue to fail. Dedicated new
regression tests preserve these cases without
weakening any previously approved baseline.

The independent necessary-geometry audit is **not** the final
full-Pareto-membership oracle and does not certify all positive-`g`
regions. Built-in configurations outside proved ranges, custom
MOPs, missing component oracles and unsupported diagnostics remain
`unverified`.

A **positive analytical membership result** confirms the stated
geometric relation under a declared tolerance, **not** a proof that
the sampled reference is an adequate finite covering. Decision
feasibility cannot be deduced from objective coordinates alone;
it is explicitly `unverified` without verified decision witnesses.
Replay is also `unverified` until separately tested.

## Explicit decision-space witness verification for original DTLZ5/6

The initial API now additionally accepts an optional
`decision_reference` matrix. It must have the **same number
of rows** as the specified objective `reference`, in exactly
the same order, with (N) decision coordinates per row.
The check is entirely opt-in; no PS or evaluation is
generated from a calibrated objective-only sidecar.

For original unmodified DTLZ5/6, independent published
decision bounds, distance functions, angles and spherical
objectives are recomputed without calling the production
MOP evaluator or sampler. The resulting
`feasibility.status` is `pass` only when every provided
decision is within bounds and produces its **aligned**
claimed objective row within the declared tolerance.
It is `fail` on invalid decision coordinates or
objective mismatches. No such result is a certificate
of global Pareto optimality (which remains a separate
global scalar-`g` question), finite covering, platform
portability or numerical backend validity.

For other MOPs, only declared decision bounds may be
checked generically; without an independently established
original equation/constraint oracle, feasibility remains
`unverified` for otherwise bounded points. An explicit
`decision_reference` and `n_points` cannot be
combined, because sampling an objective GT does not
implicitly establish row-aligned decision witnesses.
`mop.report(decision_reference=...)` inspects an
already-calibrated in-memory GT without creating,
reordering or writing any reference rows.

## Strict differential local dominance (zero-gap negative evidence)

An additional independent, **opt-in** diagnostic is available
for original DTLZ5/6 through
`mop.assess_gt(reference=F, decision_reference=X,
local_descent=True)` or the corresponding report call.

The exact mathematical theorem (Section 3.6.11 in
[`docs/gt.md`](../gt.md)) states that if a feasible
one-sided decision-position path `u(g)=u0+(g-g0)*v`
has **strictly positive derivatives in every objective**
at `g0>0`, then sufficiently small **decreases**
of `g` strictly improve all objectives. This
establishes the *existence* of a local dominator
without assuming any fixed positive `g` gap.
One-sided derivative constraints explicitly preserve
position bounds for points lying at `u_i=0`
or `u_i=1`.

The new independent analytic derivative implementation
searches feasible directions by a linear program and
reconstructs an actual feasible original-decision and
objective witness for a smaller `g`. This provides
**negative scientific evidence** for a dominated GT
candidate; it is **not** a sufficient criterion
for global Pareto membership. A missing positive
direction, solver uncertainty, or absence of a
finite witness **never** yields a positive
global certification claim. All first-order LP
and witness measurements require the accepted
method/environment/result gates before numerical
results may be promoted to certified evidence,
so `local_descent.status` remains
`unverified` for portable scientific certification.

This opt-in procedure requires aligned decision
witnesses because the objective-only GT does
not carry the original position variables.
The default GT assessment and report remain
lightweight, do not generate new references,
and do not modify original sidecars, regressions
or approved baselines.

## Analytical g=0 curve dominance witnesses

For original DTLZ5/6, `GTAssessment.curve_dominance`
records an independent check against the **complete continuous
g=0 Pareto curve** (Theorem 6 of [`docs/gt.md`](../gt.md)).
The comparison is analytical, not a discrete GT lookup or a
second call to the production sampler or evaluator.
Its result distinguishes reconstructed numerical dominance
witnesses, objectives that violate the complete-curve dominance
criterion, and tolerance-sensitive boundary cases. It records
the original row indices and witness objective vectors.

A feasible positive-`g` objective vector that is strictly
dominated by a reconstructed `g=0` curve vector is **not**
globally Pareto-optimal; numerical evidence is nevertheless
subject to the accepted mathematical, environment and
output gates before promotion to a certified failure.
Conversely, **absence of a g=0 curve dominator is never a
global Pareto pass**, because a positive-`g` feasible point
may still dominate it. As currently implemented,
`curve_dominance.status=unverified` documents this limited
scientific certification status.

This check uses objective coordinates already supplied to
`assess_gt()` and appears in `mop.report()` for original
DTLZ5/6 without an additional scalar optimization search.
It does not alter GT generation, numeric thresholds or frozen
calibration references.

## Optional DTLZ5/6 bounded-margin dominance diagnostic

For the original DTLZ5/6 only, `assess_gt()` accepts
`dominance_margin=delta` and
`dominance_max_intervals=4096`. `mop.report()` forwards
the same optional arguments. The default `None` **does not**
perform this potentially expensive numerical search.

The independent `_dtlz56_dominance_bounds` implementation tests
whether the entire continuous feasible scalar `g` interval can
exclude a feasible vector that weakly improves all coordinates and
strictly improves **at least one** by the requested positive
objective-space margin. For each `g` interval `[L,U]`, the
feasibility relaxation uses smaller radius `1+L` and a larger
angular patch from `U`. If a dominator existed anywhere in the
interval, its radial contraction would be feasible in this
relaxation; excluding the relaxed patch therefore excludes the
whole interval **under exact real arithmetic**. Interval budget
exhaustion always produces `unverified`, not a spurious pass.

The result appears in the distinct `assessment.dominance`
property. It records `numerically_excluded`, `unresolved`,
the margin and interval effort, while **the property status
remains `unverified`**: the numerical implementation uses
ordinary floating point with outward safety slack, not a
directed-rounded proof; furthermore, proving that no
`delta`-dominator exists does **not** exclude strict
improvements smaller than `delta`. This is incremental
quantitative evidence, **not** sufficient global Pareto
membership or a certification-environment gate. The normal
`validity` status for the full original `M>=4` GT
is unchanged. No official tolerance or release threshold
is established by this opt-in diagnostic.

## Exact scalar-radius dominance theorem and numerical audit

For the **original** DTLZ5/6 objective image, every feasible vector
satisfies `||f||_2 = 1 + g`. Consequently, for an independently verified
feasible point `y`, the following is an **exact necessary and sufficient
mathematical condition** for global Pareto efficiency:

```
y belongs to the true Pareto front
   iff there is no feasible z <= y with g(z) < g(y).
```

Proof: if `z <= y` with a strict improvement in any coordinate,
the nonnegative Euclidean norm decreases strictly, hence
`g(z) < g(y)`. Conversely, if `g(z) < g(y)` and `z <= y`,
the strict radius decrease ensures a strict objective improvement.
This reduces the global dominance predicate to feasibility of
objective caps over **one continuous scalar g**, independent of the
number of original decision-space distance variables.

An **optional numerical approximation**, distinct from the previous
coordinatewise-`dominance_margin` test, is exposed as

```python
audit = mop.assess_gt(
    reference=front,
    dominance_g_gap=1e-4,
    dominance_g_max_intervals=8192,
)
print(audit.global_dominance.to_dict())
mop.report(dominance_g_gap=1e-4)
```

The independent audit restricts attention to
`0 <= g <= g(y)-dominance_g_gap`. It either obtains an
independently reconstructed feasible *numerical dominance witness*,
excludes the continuous closed interval by optimistic angular/radial
feasibility relaxations (under floating-point slack), or reports the
unresolved search budget. It never silently reuses the production
scalarizer, PS/PF or evaluator. It is disabled by default.

**Do not conflate the exact mathematical equivalence with its finite
numerical evaluator.** This implementation neither excludes
improvements with `0 < g(y)-g(z) < dominance_g_gap`, nor uses
directed-rounded or verified transcendental interval bounds.
Consequently the public `global_dominance.status` remains
`unverified` for the scientific certification decision, even
when the numerical check excludes the requested gap; the
original DTLZ5/6 M>=4 `validity` status stays `unverified`.
Numerically reconstructed witnesses are reported with their
coordinates/parameters for independently checked scientific
investigation. They should be confirmed on the certified
execution environment before being accepted as a definitive
software defect.

Remaining step: establish mathematically and numerically justified
evaluation of the **zero-gap** criterion (including boundary limits
and directed-rounding uncertainties), then complete the GT
configuration, coverage and indicator-adequacy evidence.

## Quantitative finite-reference diagnostics

Generic structural checks report input shape, finite values,
actual versus requested count, exact distinct rows and duplicates.
A malformed nonfinite reference reports `validity=fail`, without
attempting undefined distance computations.

Where independent continuous-front objective probes are available,
`coverage` includes maximum and mean nearest-reference gaps
under the **original Euclidean objective-space metric**, with
seed/budget recorded. When the MOP has independently named
disconnected or separate Pareto components, report covering gaps
**by component**; an absent component has `points=0` and
unavailable, not zero, distance. `regularity` includes
nearest-neighbor mean, maximum and coefficient of variation
**within the same components**, avoiding unsupported aggregation
across intrinsic dimensions. Else it provides a whole-reference
distance diagnostic with its scope explicitly declared.

These finite-probe values remain `status=unverified` as
**acceptance judgments** until MOP/component/budget-specific
thresholds have been independently justified and approved.
An explicit negative membership, malformed coordinate,
missing known extreme/component at an eligible budget,
or duplicate contract violation produces `fail` for
the applicable structural/mathematical property.

## Tests and acceptance policy

The same computation used by the public API is used by the
**new** unit test module
`tests/unit/test_gt_assessment_contract.py`. Tests cover
independent positive analytical cases, intentionally injected
off-front values, missing known extrema/components, duplicates,
nonfinite rows, custom/subclass non-certification, stable
deterministic diagnostics, no sidecar writes and `mop.report()`
semantics. Existing historical regression tests remain unchanged.

Full scientific adequacy must ultimately be established by
a separately reviewed **MOP × configuration × GT-budget acceptance
matrix**, using independent mathematical oracles plus
geometry-specific covering/spacing criteria. Representative
objective counts, DPF intrinsic dimensions, constrained N
configurations and the existing 2,000/10,000 default budgets
belong in a bounded integration or scientific certification job
rather than making every routine unit test expensive.

Do **not** derive acceptance tolerances by copying current outputs
as expected results. A property unsupported by evidence must
be reported as `unverified`; for a required release claim, that
status blocks **certification of the claim**, not necessarily the
software's ordinary operations. A confirmed false Pareto point
or missing mandatory component is a scientific defect requiring
correction under the existing ADR/regression safeguards.

## Certification of locally generated experimental evidence

**Normative rule (maintainer approval 2026-11-23):** scientific evidence
generated on an uncertified local machine is **provisional** until its
execution environment and numerical results have passed an independent,
source-bound verification. Recording a dependency list or a Git commit
alone does not certify an execution. Conversely, running on a verified
platform does not establish the mathematical validity of an oracle.

Certification requires **three separate gates**:

1. **Mathematical/oracle gate:** use an independent published-equation
   derivation, analytic invariant, rigorous bound, or suitable trusted
   numerical reference. An oracle must not certify the GT simply by
   reusing the same `pf()`, `ps()` or evaluator implementation;
   agreement between two implementations sharing an error is not proof.
2. **Environment gate:** capture the exact source revision, Python and
   installed dependency versions, numerical backends (including relevant
   BLAS/LAPACK details), platform/architecture, algorithm/validator
   versions and numerical settings. Run the established MoeaBench
   numerical stability checks on **that execution environment** with
   approved, immutable references and tolerances. If a check is
   unavailable or fails, the environment is **unverified** and its
   outputs cannot be promoted to certified evidence.
3. **Result/provenance gate:** record MOP class/configuration, original
   GT origin, point and probe budgets, seeds, method parameters,
   geometry/normalization, thresholds and output hashes. Validate
   mathematical residuals and representative computed outputs against
   an **independent reference** (including a bounded source-bound
   cross-platform/CI check where applicable), using justified numeric
   tolerances. Report the check's outcome and limitations.

Large local experiments may be run in parallel and their outputs reused
as **provisional evidence**. They become certified only when the three
gates above pass for the **same source, environment and artifact**.
Portable containers, lockfiles and environment manifests improve
reproducibility but do not alone guarantee bitwise identical
floating-point behavior or establish platform certification. In
particular, differing BLAS kernels, floating-point reduction order
and optimization paths may alter numerical outcomes.

The certification record must distinguish `method_validated`,
`environment_verified`, and `result_verified` as separate
claims, with explicit evidence and `unverified` or `fail` when
applicable. No unpublished local computation is silently treated as
a certified GT or as authority to replace frozen baselines.

This rule governs the forthcoming scientific certification pipeline;
**it does not assert that the current `GTAssessment` API already
implements all three gates**. Implementation and artifact-schema
acceptance are tracked under [#175](https://github.com/monacofj/moeabench/issues/175)
and [superissue #176](https://github.com/monacofj/moeabench/issues/176).

## Relationship to existing accepted decisions

- ADR 0006 governs original MOP mathematical PS/PF.
- ADR 0018 preserves 2,000 DTLZ / 10,000 DPF GT defaults;
  this ADR introduces no automatic change to those budgets.
- ADR 0056 governs numerical/performance certificates, frozen
  historical references and approved migrations.
- ADR 0057 governs MOP-owned generation and explicit GT calibration;
  assessment is read-only and never replaces calibration.
- ADR 0058 governs the continuous-certified or finite-sampled
  absolute HV denominator; a GT assessment **does not by itself**
  certify a new exact continuous Hypervolume formula.

Accepted architectural changes and revisions of existing
regression tests still require prior maintainer authorization
where applicable. No historical baseline, certificate or
previously accepted test may be rewritten just to pass this
assessment. The current implementation contract is specified
in [`docs/gt.md`](../gt.md); historical experiments belong in
[`docs/research/`](../research/gt-issue119/).

## Explicitly remaining work

1. Independent **sufficient** global Pareto-membership certification
   for the original DTLZ5/6 positive-`g` boundary (necessary radial
   geometry and exact `g=0` membership are now measured), and
   configuration-specific evidence beyond the cases checked.
2. Decision-space feasibility for **all remaining MOPs and constraints**, verified replay, and source provenance at the assessed-reference level. Aligned original DTLZ5/6 decision-to-objective witnesses are now independently inspected; this does not establish overall GT certification.
3. Approved per-MOP/per-component quantitative resolution and
   regularity thresholds, and comparison with the original
   2,000/10,000 calibration budgets.
4. Complete source-bound release-readiness/certification matrix;
   current results do not establish a release certificate.

These items remain open scientific qualification work, **not**
unimplemented parts of the already available inspection API.
