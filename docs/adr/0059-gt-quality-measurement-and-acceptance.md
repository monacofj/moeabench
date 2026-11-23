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
2. Decision-space feasibility witnesses, verified replay and
   source provenance at the assessed-reference level.
3. Approved per-MOP/per-component quantitative resolution and
   regularity thresholds, and comparison with the original
   2,000/10,000 calibration budgets.
4. Complete source-bound release-readiness/certification matrix;
   current results do not establish a release certificate.

These items remain open scientific qualification work, **not**
unimplemented parts of the already available inspection API.
